"""
Collection of GPU-accelerated K-Nearest Neighbors backends (FAISS-based grid/Voronoi caches,
PyKeOps, PyTorch3D, cuML) sharing a build()/query() interface, used to find the nearest
Gaussians to each queried coordinate when evaluating a GaussianIntensityField.
"""

import os
import sys
from typing import Callable, Literal, Union, Optional

import torch
# from pytorch3d.ops import knn_points
# import cuml
# from cuml.neighbors import NearestNeighbors
import faiss.contrib.torch_utils  # Allows FAISS to read PyTorch GPU tensors with zero-copy
import faiss


class StructuredVoronoiFAISSKNN:
    """
    A high-performance K-Nearest Neighbors cache using a regular grid of control points,
    backed by a FAISS GPU IVFFlat Index for lighting-fast candidate pool generation.
    """

    def __init__(
            self,
            k: int,
            range_min: Union[float, torch.Tensor],
            range_max: Union[float, torch.Tensor],
            spacing: Union[float, torch.Tensor],
            k_candidates: Optional[int],
            nlist: int = 256,
            nprobe: int = 16,
            dim: Optional[int] = None,
            device: Optional[str] = None,
            metric: Literal["l2", "mahalanobis"] = "l2",
    ):
        """
        Args:
            range_min, range_max, spacing, k, k_candidates, dim, device: Same as original.
            nlist (int): Number of Voronoi cells (clusters) for the FAISS index to train.
            nprobe (int): Number of adjacent Voronoi cells to search during candidate pooling.
        """
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cpu':
            raise ValueError("StructuredGridFAISSKNN requires a CUDA device.")

        if metric not in ("l2", "mahalanobis"):
            raise ValueError(f"Unknown metric {metric}")

        self.k = k
        if k_candidates is not None:
            if k_candidates < k:
                raise ValueError(f"k_candidates ({k_candidates}) must be >= k ({k}).")
            elif k_candidates == k:
                k_candidates = None
        self.k_candidates = k_candidates

        self.metric = metric

        # Infer dimensionality
        tensors = [t for t in (range_min, range_max, spacing) if isinstance(t, torch.Tensor) and t.numel() > 1]
        if tensors:
            self.D = tensors[0].numel()
        else:
            if dim is None:
                raise ValueError("Dimensionality 'dim' must be provided if min, max, and spacing are scalars.")
            self.D = dim

        self.range_min = self._to_tensor(range_min, self.D, self.device)
        self.range_max = self._to_tensor(range_max, self.D, self.device)
        self.spacing = self._to_tensor(spacing, self.D, self.device)

        self.grid_shape = torch.round((self.range_max - self.range_min) / self.spacing).to(torch.long) + 1

        self.strides = torch.tensor(
            [self.grid_shape[i + 1:].prod() if i + 1 < self.D else 1 for i in range(self.D)],
            dtype=torch.long,
            device=self.device
        )

        # Precompute the control points coordinates (G, D) tensor once during initialization
        axes = [
            torch.arange(self.grid_shape[i], device=self.device, dtype=torch.float32) * self.spacing[i] +
            self.range_min[i]
            for i in range(self.D)
        ]
        self.control_points = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).view(-1, self.D).contiguous()

        self.cached_knn_indices: Optional[torch.Tensor] = None
        self.target_points: Optional[torch.Tensor] = None
        self.precisions: torch.Tensor | None = None

        # --- FAISS GPU Initialization ---
        self.res = faiss.StandardGpuResources()
        # Initialize IVFFlat on GPU (METRIC_L2 computes squared Euclidean distance)
        self.faiss_index = faiss.GpuIndexIVFFlat(self.res, self.D, nlist, faiss.METRIC_L2)
        self.faiss_index.nprobe = nprobe
        self.is_trained = False

    def _to_tensor(self, val: Union[float, torch.Tensor], dim: int, device: str) -> torch.Tensor:
        """Helper method to convert scalars or tensors into a standard (D,) tensor."""
        if isinstance(val, (float, int)):
            return torch.full((dim,), float(val), dtype=torch.float32, device=device)
        elif isinstance(val, torch.Tensor):
            val = val.to(device).contiguous()
            if val.numel() == 1:
                return torch.full((dim,), float(val.item()), dtype=torch.float32, device=device)
            elif val.numel() == dim:
                return val.view(dim).to(torch.float32)
            else:
                raise ValueError(f"Tensor shape mismatch. Expected {dim} elements, got {val.numel()}")
        else:
            raise TypeError("Input must be a float or torch.Tensor")

    @torch.no_grad()
    def build(self, target_points: torch.Tensor, *args, precisions: Callable[[], torch.Tensor] | None = None) -> None:
        """
        Populates the control points' candidate pools using FAISS GPU clustering.
        """
        # FAISS requires contiguous float32 PyTorch tensors for zero-copy
        target_points = target_points.to(self.device)
        if target_points.dtype != torch.float32:
            target_points = target_points.to(torch.float32)
        target_points = target_points.detach().contiguous()

        # 1. Train the FAISS index ONCE
        if not self.is_trained:
            self.faiss_index.train(target_points)
            self.is_trained = True

        # 2. Update the spatial index with the new target locations
        self.faiss_index.reset()
        self.faiss_index.add(target_points)

        # 3. Query FAISS with all control points simultaneously to get the candidate pools
        k_search = self.k_candidates if self.k_candidates is not None else self.k

        # faiss.contrib.torch_utils intercepts this and returns PyTorch tensors
        _, self.cached_knn_indices = self.faiss_index.search(self.control_points, k=k_search)

        # Sync CUDA streams to ensure FAISS finishes writing before PyTorch proceeds
        torch.cuda.synchronize(self.device)

        # Save actual points in memory for the reranking stage in query()
        self.target_points = target_points

        if self.metric == "mahalanobis":
            if precisions is None:
                raise ValueError("metric mahalanobis requires precision matrices")

            self.precisions = precisions().detach().to(self.device, dtype=torch.float32).contiguous()

    def _distances(self, query_coordinates: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        difference = query_coordinates.unsqueeze(1) - self.target_points[candidates]

        if self.metric == "mahalanobis":
            return torch.einsum("nki,nkij,nkj->nk", difference, self.precisions[candidates], difference)

        return (difference ** 2).sum(dim=-1)


    @torch.no_grad()
    def query(self, query_coords: torch.Tensor, *args) -> torch.Tensor:
        """
        Retrieves the KNN indices using O(1) grid quantization, then computes
        exact distances locally to rerank the candidates.
        """
        if self.cached_knn_indices is None or self.target_points is None:
            raise RuntimeError("You must call .build(target_points) before querying.")

        query_coords = query_coords.to(self.device)

        # 1. Quantize
        grid_idx = torch.round((query_coords - self.range_min) / self.spacing).to(torch.long)

        # 2. Clamp
        max_bounds = self.grid_shape - 1
        grid_idx = torch.clamp(torch.clamp(grid_idx, min=0), max=max_bounds)

        # 3. Flatten to 1D index
        linear_idx = (grid_idx * self.strides).sum(dim=1)

        # 4. Fetch the search_k candidates
        retrieved_indices = self.cached_knn_indices[linear_idx]

        # 5. Fast path: return directly if no reranking is needed
        if self.k_candidates is None:
            return retrieved_indices

        # Compute exact squared distances from query coords to their localized candidates
        sq_dists = self._distances(query_coords, retrieved_indices)

        # Find the local indices of the top k closest candidates
        _, local_top_k_indices = torch.topk(sq_dists, k=self.k, dim=1, largest=False, sorted=True)

        # Gather the final indices mapping back to the original target_points tensor
        final_indices = torch.gather(retrieved_indices, 1, local_top_k_indices)

        return final_indices

    @torch.no_grad()
    def query_via_chunks(self, query_coords: torch.Tensor, batch_size: int = 1_000_000) -> torch.Tensor:
        """
        Queries massive coordinate tensors in batches to prevent VRAM exhaustion
        during intermediate distance calculation and candidate reranking.

        Args:
            query_coords (torch.Tensor): Coordinates to find neighbors for.
                Expected shape: (M, D).
            batch_size (int): Number of queries to process simultaneously.

        Returns:
            torch.Tensor: Indices of the k nearest neighbors.
                Expected shape: (M, k).
        """
        if self.cached_knn_indices is None or self.target_points is None:
            raise RuntimeError("You must call .build(target_points) before querying.")

        if batch_size is not None and self.metric == "mahalanobis":
            batch_size //= 4  # eyeballed, account for larger tensors for mahalanobis, so it doesn't distort the VRAM measurements

        query_coords = query_coords.to(self.device)
        if query_coords.dtype != torch.float32:
            query_coords = query_coords.to(torch.float32)

        M = query_coords.shape[0]

        # Pre-allocate exact output shape
        all_indices = torch.empty((M, self.k), dtype=torch.int64, device=self.device)

        # Cache max bounds to avoid repeated computations inside the loop
        max_bounds = self.grid_shape - 1

        for i in range(0, M, batch_size):
            end_idx = min(i + batch_size, M)
            batch_coords = query_coords[i:end_idx].contiguous()

            # 1. Quantize
            grid_idx = torch.round((batch_coords - self.range_min) / self.spacing).to(torch.long)

            # 2. Clamp
            grid_idx = torch.clamp(torch.clamp(grid_idx, min=0), max=max_bounds)

            # 3. Flatten to 1D index
            linear_idx = (grid_idx * self.strides).sum(dim=1)

            # 4. Fetch the search_k candidates
            retrieved_indices = self.cached_knn_indices[linear_idx]

            # 5. Fast path: attach directly if no reranking is needed
            if self.k_candidates is None:
                all_indices[i:end_idx] = retrieved_indices
                continue

            # Compute exact squared distances from query coords to their localized candidates
            sq_dists = self._distances(batch_coords, retrieved_indices)

            # Find the local indices of the top k closest candidates
            _, local_top_k_indices = torch.topk(sq_dists, k=self.k, dim=1, largest=False, sorted=True)

            # Gather the final indices mapping back to the original target_points tensor
            final_indices = torch.gather(retrieved_indices, 1, local_top_k_indices)

            all_indices[i:end_idx] = final_indices

        return all_indices


class VoronoiFAISSKNN:
    """
    A direct K-Nearest Neighbors index using FAISS GPU IVFFlat.
    Coordinates query the target points directly through Voronoi cells,
    completely bypassing intermediate spatial grids.
    """

    def __init__(
            self,
            k: int,
            dim: int = 3,
            nlist: int = 2048,
            nprobe: int = 64,
            device: Optional[Union[str, torch.device]] = None,
            **kwargs,
    ):
        """
        Args:
            k (int): Number of nearest neighbors to retrieve.
            dim (int): Dimensionality of the coordinates (typically 3).
            nlist (int): Number of Voronoi cells (clusters) to partition the space into.
            nprobe (int): Number of adjacent Voronoi cells to search per query.
            device: The CUDA device to store the index and execute on.
        """
        self.k = k
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if self.device.type != 'cuda':
            raise ValueError("DirectFAISSKNN requires a CUDA device for GPU acceleration.")

        # Initialize FAISS GPU Resources
        self.res = faiss.StandardGpuResources()

        # Create the Inverted File Flat (IVFFlat) index on the GPU.
        # METRIC_L2 means squared Euclidean distance.
        self.index = faiss.GpuIndexIVFFlat(self.res, self.dim, self.nlist, faiss.METRIC_L2)
        self.index.nprobe = self.nprobe

        self.is_trained = False
        self.target_points: Optional[torch.Tensor] = None

    @torch.no_grad()
    def build(self, target_points: torch.Tensor) -> None:
        """
        Trains the Voronoi centroids (only on the first call) and maps the
        target points into their respective clusters.
        """
        # FAISS requires contiguous 32-bit floats
        target_points = target_points.to(self.device)
        if target_points.dtype != torch.float32:
            target_points = target_points.to(torch.float32)
        target_points = target_points.detach().contiguous()

        # 1. Train K-Means centroids ONCE.
        # This defines the structural scaffolding of the space.
        if not self.is_trained:
            self.index.train(target_points)
            self.is_trained = True

        # 2. Clear old data and assign the new target points to the Voronoi cells
        self.index.reset()
        self.index.add(target_points)

        self.target_points = target_points

    @torch.no_grad()
    def query(self, query_coords: torch.Tensor) -> torch.Tensor:
        """
        Performs a direct FAISS search for all query coordinates.
        Note: Passing 30 million queries directly might OOM the GPU memory bus
        depending on VRAM limits. Use query_via_chunks if this occurs.
        """
        if not self.is_trained:
            raise RuntimeError("You must call .build(target_points) before querying.")

        query_coords = query_coords.to(self.device)
        if query_coords.dtype != torch.float32:
            query_coords = query_coords.to(torch.float32)
        query_coords = query_coords.contiguous()

        # faiss.contrib.torch_utils intercepts the PyTorch tensors natively
        distances, indices = self.index.search(query_coords, self.k)

        return indices

    @torch.no_grad()
    def query_via_chunks(self, query_coords: torch.Tensor, batch_size: int = 1_000_000) -> torch.Tensor:
        """
        Queries massive coordinate tensors in batches to prevent VRAM exhaustion
        during FAISS internal result allocation.
        """
        if not self.is_trained:
            raise RuntimeError("You must call .build(target_points) before querying.")

        query_coords = query_coords.to(self.device)
        if query_coords.dtype != torch.float32:
            query_coords = query_coords.to(torch.float32)

        M = query_coords.shape[0]

        # Pre-allocate exact output shape
        all_indices = torch.empty((M, self.k), dtype=torch.int64, device=self.device)

        for i in range(0, M, batch_size):
            end_idx = min(i + batch_size, M)
            batch_coords = query_coords[i:end_idx].contiguous()

            # Execute chunked query
            _, batch_indices = self.index.search(batch_coords, self.k)

            all_indices[i:end_idx] = batch_indices

        return all_indices

class FaissKNN:
    """
    A GPU-accelerated K-Nearest Neighbors wrapper using FAISS.
    Accepts and returns PyTorch tensors directly.
    """

    def __init__(self, k: int, dim: int, device: torch.device, **kwargs):
        """
        Initializes the FAISS GPU Index.

        Args:
            k (int): Number of nearest neighbors to retrieve.
            dim (int): Dimensionality of the coordinates (e.g., 3).
            device (torch.device): The device to store the index and run queries on.
        """
        self.k = k
        self.dim = dim
        self.device = device

        res = faiss.StandardGpuResources()  # Use GPU resources
        cpu_index = faiss.IndexFlatL2(self.dim)  # Use simple L2 distance
        self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # Move index to GPU 0

    def build(self, target_points: torch.Tensor) -> None:
        """
        Populates the FAISS index with the target points (Gaussian means).

        Args:
            target_points (torch.Tensor): The coordinates to build the index over.
                Expected shape: (N, dim).
        """
        assert target_points.shape[1] == self.dim
        target_points = target_points.detach().contiguous()

        self.index.reset()
        self.index.add(target_points)

    def query(self, query_coords: torch.Tensor) -> torch.Tensor:
        """
        Queries the nearest neighbors for the given coordinates.

        Args:
            query_coords (torch.Tensor): Coordinates to find neighbors for.
                Expected shape: (M, dim).

        Returns:
            torch.Tensor: Indices of the k nearest neighbors.
                Expected shape: (M, k).
        """

        query_coords = query_coords.detach().contiguous()
        a, indices = self.index.search(query_coords, self.k)
        return indices

    @torch.no_grad()
    def query_via_chunks(self, query_coords: torch.Tensor, batch_size: int = 1_000_000) -> torch.Tensor:
        """
        Queries massive coordinate tensors in batches to prevent VRAM exhaustion
        during FAISS internal result allocation.
        """

        query_coords = query_coords.to(self.device)
        if query_coords.dtype != torch.float32:
            query_coords = query_coords.to(torch.float32)

        M = query_coords.shape[0]

        # Pre-allocate exact output shape
        all_indices = torch.empty((M, self.k), dtype=torch.int64, device=self.device)

        for i in range(0, M, batch_size):
            end_idx = min(i + batch_size, M)
            batch_coords = query_coords[i:end_idx].contiguous()

            # Execute chunked query
            _, batch_indices = self.index.search(batch_coords, self.k)

            all_indices[i:end_idx] = batch_indices

        return all_indices


# def configure_robust_compiler_paths() -> None:
#     """
#     Forces the C++ compiler and linker to recognize Conda's nested CUDA directories.
#     Bypasses PyKeOps's hardcoded include paths by dynamically resolving the active
#     Python environment prefix.
#     """
#     # Fallback to sys.prefix if CONDA_PREFIX is not injected by the shell/IDE
#     conda_prefix: str = os.environ.get("CONDA_PREFIX", sys.prefix)
#
#     if not conda_prefix or not os.path.isdir(conda_prefix):
#         raise EnvironmentError(f"Resolved Conda prefix is invalid or missing: {conda_prefix}")
#
#     os.environ["CUDA_PATH"] = conda_prefix
#
#     # 1. Resolve deep header paths for the C++ Compiler (g++)
#     base_include: str = os.path.join(conda_prefix, "include")
#     target_include: str = os.path.join(conda_prefix, "targets", "x86_64-linux", "include")
#
#     c_includes: str = f"{base_include}:{target_include}"
#     os.environ["CPATH"] = f"{c_includes}:{os.environ.get('CPATH', '')}".strip(":")
#     os.environ["CPLUS_INCLUDE_PATH"] = f"{c_includes}:{os.environ.get('CPLUS_INCLUDE_PATH', '')}".strip(":")
#
#     # 2. Re-inject the linker paths
#     base_lib: str = os.path.join(conda_prefix, "lib")
#     stubs_lib: str = os.path.join(conda_prefix, "targets", "x86_64-linux", "lib", "stubs")
#     os.environ["LD_LIBRARY_PATH"] = f"{base_lib}:{stubs_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}".strip(":")

# # Inject paths BEFORE PyKeOps import
# configure_robust_compiler_paths()
# import pykeops
# from pykeops.torch import LazyTensor
# pykeops.set_verbose(True)
# pykeops.clean_pykeops()

class StructuredGridKNN:
    """
    A purely structured K-Nearest Neighbors cache with candidate pool reranking.
    It generates a regular grid of control points, precomputes a large pool of K-NN (search_k),
    and at query time, reranks these candidates based on the exact query coordinate to return the top K.
    """

    def __init__(
            self,
            range_min: Union[float, torch.Tensor],
            range_max: Union[float, torch.Tensor],
            spacing: Union[float, torch.Tensor],
            k: int,
            k_candidates: Optional[int],
            dim: Optional[int] = None,
            device: Optional['str'] = None,
            **kwargs
    ):
        """
        Initializes the coordinate system constraints and grid logic.

        Args:
            range_min (Union[float, torch.Tensor]): The minimum bounding coordinates of the system.
                If scalar, applied to all dimensions.
                Expected shape if tensor: (D,) where D is the dimensionality.
            range_max (Union[float, torch.Tensor]): The maximum bounding coordinates of the system.
                If scalar, applied to all dimensions.
                Expected shape if tensor: (D,).
            spacing (Union[float, torch.Tensor]): The distance between control points along each axis.
                If scalar, applied to all dimensions.
                Expected shape if tensor: (D,).
            k (int): The final number of nearest neighbors to return during query().
            k_candidates (int): The larger pool of nearest neighbors to compute and cache during build().
            dim (Optional[int]): The dimensionality of the coordinate system. Required if
                range_min, range_max, and spacing are all scalars.
            device (torch.device): The device to store the cache on.
        """

        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        if k_candidates is not None:
            if k_candidates < k:
                raise ValueError(f"search_k ({k_candidates}) must be greater than or equal to k ({k}).")
            elif k_candidates == k:
                k_candidates = None
        self.k_candidates = k_candidates

        # Infer dimensionality if not explicitly provided
        tensors = [t for t in (range_min, range_max, spacing) if isinstance(t, torch.Tensor) and t.numel() > 1]
        if tensors:
            inferred_dim = tensors[0].numel()
            if dim is not None and dim != inferred_dim:
                raise ValueError(f"Explicit dim {dim} contradicts inferred tensor dim {inferred_dim}.")
            self.D = inferred_dim
        else:
            if dim is None:
                raise ValueError("Dimensionality 'dim' must be provided if min, max, and spacing are all scalars.")
            self.D = dim

        self.range_min = self._to_tensor(range_min, self.D, self.device)
        self.range_max = self._to_tensor(range_max, self.D, self.device)
        self.spacing = self._to_tensor(spacing, self.D, self.device)

        self.grid_shape = torch.round((self.range_max - self.range_min) / self.spacing).to(torch.long) + 1

        self.strides = torch.tensor(
            [self.grid_shape[i + 1:].prod() if i + 1 < self.D else 1 for i in range(self.D)],
            dtype=torch.long,
            device=self.device
        )

        self.cached_knn_indices: Optional[torch.Tensor] = None
        self.target_points: Optional[torch.Tensor] = None

    def _to_tensor(self, val: Union[float, torch.Tensor], dim: int, device: torch.device) -> torch.Tensor:
        """
        Helper method to convert scalars or tensors into a standard (D,) tensor.
        """
        if isinstance(val, (float, int)):
            return torch.full((dim,), float(val), dtype=torch.float32, device=device)
        elif isinstance(val, torch.Tensor):
            val = val.to(device).contiguous()
            if val.numel() == 1:
                return torch.full((dim,), float(val.item()), dtype=torch.float32, device=device)
            elif val.numel() == dim:
                return val.view(dim).to(torch.float32)
            else:
                raise ValueError(f"Tensor shape mismatch. Expected {dim} elements, got {val.numel()}")
        else:
            raise TypeError("Input must be a float or torch.Tensor")

    def build(self, target_points: torch.Tensor, *args) -> None:
        """
        Generates the structured control points and computes their search_k neighbors against the target data.

        Args:
            target_points (torch.Tensor): The actual dataset to search against.
                Expected shape: (N, D) where N is the number of target points.
        """
        # Kept in memory to allow exact distance reranking during query()
        self.target_points = target_points.to(self.device).contiguous()

        axes = [
            torch.arange(self.grid_shape[i], device=self.device) * self.spacing[i] + self.range_min[i]
            for i in range(self.D)
        ]

        control_points = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).view(-1, self.D)

        c_i = LazyTensor(control_points.unsqueeze(1))
        t_j = LazyTensor(self.target_points.unsqueeze(0))

        D_ij = ((c_i - t_j) ** 2).sum(-1)

        # Cache search_k indices
        # Shape: (N_grid, search_k)
        k = self.k_candidates if self.k_candidates is not None else self.k
        self.cached_knn_indices = D_ij.argKmin(K=k, dim=1)

    def query(self, query_coords: torch.Tensor, *args) -> torch.Tensor:
        """
        Retrieves the KNN indices for arbitrary coordinates using O(1) grid quantization.
        If search_k > k, reranks the retrieved candidates based on exact distance to query_coords.

        Args:
            query_coords (torch.Tensor): The coordinates to query.
                Expected shape: (M, D) where M is the batch size and D is the dimensionality.

        Returns:
            torch.Tensor: The indices of the K-nearest neighbors in the target dataset.
                Expected shape: (M, k).

        Raises:
            RuntimeError: If called before build().
        """
        if self.cached_knn_indices is None or self.target_points is None:
            raise RuntimeError("You must call .build(target_points) before querying.")

        query_coords = query_coords.to(self.device)

        # 1. Quantize
        grid_idx = torch.round((query_coords - self.range_min) / self.spacing).to(torch.long)

        # 2. Clamp using split calls for safety across PyTorch versions
        max_bounds = self.grid_shape - 1
        grid_idx = torch.clamp(torch.clamp(grid_idx, min=0), max=max_bounds)

        # 3. Flatten to 1D index
        linear_idx = (grid_idx * self.strides).sum(dim=1)

        # 4. Fetch the search_k candidates
        # Shape: (M, search_k)
        retrieved_indices = self.cached_knn_indices[linear_idx]

        # 5. Fast path: return directly if no reranking is needed
        if self.k_candidates is None:
            return retrieved_indices

        # 6. Reranking path
        # Fetch the actual coordinates of the candidates
        # Shape: (M, search_k, D)
        candidate_coords = self.target_points[retrieved_indices]

        # Compute exact squared distances from the actual query coordinates to the candidates
        # Shape: (M, search_k)
        sq_dists = ((query_coords.unsqueeze(1) - candidate_coords) ** 2).sum(dim=-1)

        # Find the local indices of the top k closest candidates
        # Shape: (M, k)
        _, local_top_k_indices = torch.topk(sq_dists, k=self.k, dim=1, largest=False, sorted=True)

        # Gather the final indices mapping back to the original target_points tensor
        # Shape: (M, k)
        final_indices = torch.gather(retrieved_indices, 1, local_top_k_indices)

        return final_indices


class ExactKeOpsKNN:
    """
    A direct, exact K-Nearest Neighbors compute engine using PyKeOps.
    Performs lazy evaluation of the L2 distance matrix to avoid VRAM OOM,
    but executes a full brute-force search.
    """

    def __init__(self, k: int, device: torch.device):
        """
        Initializes the KeOps KNN wrapper.

        Args:
            k (int): Number of nearest neighbors to retrieve.
            device (torch.device): The device to store the targets and run queries on.
        """
        self.k = k
        self.device = device
        self.target_points: Optional[torch.Tensor] = None

    def build(self, target_points: torch.Tensor) -> None:
        """
        Registers the target points for the KNN search.
        Unlike FAISS, KeOps does not build an index; it simply stores the data for query-time compute.

        Args:
            target_points (torch.Tensor): The coordinates to search against.
                Expected shape: (N, dim).
        """
        self.target_points = target_points.to(self.device).contiguous()

    def query(self, query_coords: torch.Tensor, batch_size: int = 1000000) -> torch.Tensor:
        """
        Queries the exact K-nearest neighbors for the given coordinates using lazy evaluation.

        Args:
            query_coords (torch.Tensor): Coordinates to find neighbors for.
                Expected shape: (M, dim).
            batch_size (int): Number of queries to process per KeOps call to avoid timeout/indexing limits.

        Returns:
            torch.Tensor: Indices of the k nearest neighbors in the target dataset.
                Expected shape: (M, k).
        """
        if self.target_points is None:
            raise RuntimeError("You must call .build(target_points) before querying.")

        query_coords = query_coords.to(self.device).contiguous()
        M, dim = query_coords.shape

        # Pre-allocate output tensor
        all_indices = torch.empty((M, self.k), dtype=torch.int64, device=self.device)

        # y_j represents the target points (1, N, dim)
        y_j = LazyTensor(self.target_points.unsqueeze(0))

        for i in range(0, M, batch_size):
            batch_coords = query_coords[i:i + batch_size]

            # x_i represents the query points (batch_size, 1, dim)
            x_i = LazyTensor(batch_coords.unsqueeze(1))

            # Symbolic definition of the squared L2 distance
            # No actual matrix is allocated here
            D_ij = ((x_i - y_j) ** 2).sum(-1)

            # KeOps executes the reduction (argKmin) block by block on the GPU
            batch_indices = D_ij.argKmin(K=self.k, dim=1)

            all_indices[i:i + batch_size] = batch_indices

        return all_indices



class Pytorch3DKNN:
    """
    """

    def __init__(self, k: int, dim: int, device: torch.device):
        """

        Args:
            k (int): Number of nearest neighbors to retrieve.
            dim (int): Dimensionality of the coordinates.
            device (torch.device): The device to store the index and run queries on.
        """
        self.k = k
        self.dim = dim
        self.device = device
        self.target_points: Optional[torch.Tensor] = None

    @torch.no_grad()
    def build(self, target_points: torch.Tensor) -> None:
        """
        Cache target points

        Args:
            target_points (torch.Tensor): The coordinates to build the index over.
                Expected shape: (N, dim).
        """
        self.target_points = target_points

    @torch.no_grad()
    def query(self, query_coords: torch.Tensor) -> torch.Tensor:
        """
        Queries the nearest neighbors for the given coordinates in batches.

        Args:
            query_coords (torch.Tensor): Coordinates to find neighbors for.
                Expected shape: (M, dim).

        Returns:
            torch.Tensor: Indices of the k nearest neighbors.
                Expected shape: (M, k).
        """
        # query_coords_subset: (1, M, 3)
        # centers: (1, N, 3)
        # It dynamically handles the spatial structure in heavily optimized C++/CUDA
        _, indices, _ = knn_points(
            query_coords.unsqueeze(0),
            self.target_points.unsqueeze(0),
            K=self.k
        )
        return indices.squeeze(0)

    def query_via_chunks(self, query_coords: torch.Tensor, batch_size: int = 500_000) -> torch.Tensor:
        """
        Queries the nearest neighbors for the given coordinates in chunks to prevent OOM.

        Args:
            query_coords (torch.Tensor): Coordinates to find neighbors for.
                Expected shape: (M, dim).
            batch_size (int): Number of queries to process simultaneously.

        Returns:
            torch.Tensor: Indices of the k nearest neighbors.
                Expected shape: (M, k).
        """
        if self.target_points is None:
            raise RuntimeError("You must call .build(target_points) before querying.")

        query_coords = query_coords.to(self.device).contiguous()
        M = query_coords.shape[0]

        # Pre-allocate output tensor on the target device
        all_indices = torch.empty((M, self.k), dtype=torch.int64, device=self.device)

        # PyTorch3D requires a batch dimension: (1, N, dim)
        targets_batched = self.target_points.unsqueeze(0)

        for i in range(0, M, batch_size):
            batch_coords = query_coords[i:i + batch_size]

            # batch_coords_batched: (1, batch_size, dim)
            _, batch_indices, _ = knn_points(
                batch_coords.unsqueeze(0),
                targets_batched,
                K=self.k
            )

            # Remove the batch dimension from the output: (batch_size, k)
            all_indices[i:i + batch_size] = batch_indices.squeeze(0)

        return all_indices


class CuML3DKNN:
    """
    A high-performance KNN index using RAPIDS cuML, engineered for massive 3D point clouds.
    """

    def __init__(self, k: int, dim: int, device: torch.device):
        """
        Initializes the cuML KNN index structure.

        Args:
            k (int): Number of nearest neighbors to retrieve.
            dim (int): Dimensionality of the coordinates.
            device (torch.device): The device to store the index and run queries on.
        """
        self.k = k
        self.dim = dim
        self.device = device
        self.index: Optional[NearestNeighbors] = None

    @torch.no_grad()
    def build(self, target_points: torch.Tensor) -> None:
        """
        Constructs the spatial index in VRAM.

        Unlike PyTorch3D which merely caches the tensor, this performs the heavy
        lifting of building a Random Ball Cover (rbc) index to make subsequent
        queries operate in O(N log M) instead of O(N * M).

        Args:
            target_points (torch.Tensor): The coordinates to build the index over.
                Expected shape: (N, dim).
        """
        # cuML requires contiguous 32-bit floats for maximum hardware throughput
        if target_points.dtype != torch.float32:
            target_points = target_points.to(torch.float32)
        if not target_points.is_contiguous():
            target_points = target_points.contiguous()

        # output_type='cupy' ensures cuML returns CuPy arrays instead of NumPy arrays,
        # preventing the GPU from transferring data back to host CPU RAM.
        self.index = NearestNeighbors(
            n_neighbors=self.k,
            algorithm='rbc',
            metric='euclidean',
            output_type='cupy'
        )

        # This actively builds the index on the GPU
        self.index.fit(target_points)

    @torch.no_grad()
    def query(self, query_coords: torch.Tensor) -> torch.Tensor:
        """
        Queries the nearest neighbors for the given coordinates in a single pass.

        Args:
            query_coords (torch.Tensor): Coordinates to find neighbors for.
                Expected shape: (M, dim).

        Returns:
            torch.Tensor: Indices of the k nearest neighbors.
                Expected shape: (M, k).
        """
        if self.index is None:
            raise RuntimeError("You must call .build(target_points) before querying.")

        if query_coords.dtype != torch.float32:
            query_coords = query_coords.to(torch.float32)
        if not query_coords.is_contiguous():
            query_coords = query_coords.contiguous()

        # kneighbors returns CuPy arrays due to output_type='cupy'
        distances_cp, indices_cp = self.index.kneighbors(query_coords)

        # Zero-copy cast from CuPy back to PyTorch Tensor using DLPack under the hood
        indices_pt = torch.as_tensor(indices_cp, device=self.device)
        return indices_pt

    @torch.no_grad()
    def query_via_chunks(self, query_coords: torch.Tensor, batch_size: int = 500_000) -> torch.Tensor:
        """
        Queries the nearest neighbors for the given coordinates in chunks to prevent
        VRAM exhaustion during the intermediate distance matrix allocations.

        Args:
            query_coords (torch.Tensor): Coordinates to find neighbors for.
                Expected shape: (M, dim).
            batch_size (int): Number of queries to process simultaneously.

        Returns:
            torch.Tensor: Indices of the k nearest neighbors.
                Expected shape: (M, k).
        """
        if self.index is None:
            raise RuntimeError("You must call .build(target_points) before querying.")

        query_coords = query_coords.to(self.device)
        if query_coords.dtype != torch.float32:
            query_coords = query_coords.to(torch.float32)

        M = query_coords.shape[0]

        # Pre-allocate output tensor on the target device
        all_indices = torch.empty((M, self.k), dtype=torch.int64, device=self.device)

        for i in range(0, M, batch_size):
            end_idx = min(i + batch_size, M)
            batch_coords = query_coords[i:end_idx]

            if not batch_coords.is_contiguous():
                batch_coords = batch_coords.contiguous()

            # Execute cuML query for the chunk
            _, batch_indices_cp = self.index.kneighbors(batch_coords)

            # Zero-copy assignment back into the PyTorch result tensor
            all_indices[i:end_idx] = torch.as_tensor(batch_indices_cp, device=self.device)

        return all_indices
