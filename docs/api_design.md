Vektor.__init__(config: VektorConfig = None)

Vektor.create_collection(
    name: str, dim: int, metric: str,
    mode: str = "beginner",
    M: Optional[int] = None,
    ef_construction: Optional[int] = None,
) -> Collection
    Raises: CollectionAlreadyExistsError, VektorConfigError, InvalidDimensionError

Vektor.get_collection(name: str) -> Collection
    Raises: CollectionNotFoundError

Vektor.delete_collection(name: str) -> None
    Raises: CollectionNotFoundError

Vektor.list_collections() -> list[str]

Collection.insert(id: str, vector, metadata: dict = None) -> None
    Raises: DuplicateIDError, InvalidVectorDimensionError, NonFiniteVectorError, ...

Collection.batch_insert(records: list[dict]) -> BatchInsertResult

Collection.search(
    query, k: int = 10, ef: Optional[int] = None,
    filters: Optional[dict] = None,
    strategy: str = "post",
    include_vectors: bool = False,
) -> list[SearchResult]
    Raises: InvalidEFError, EmptyCollectionError, NotImplementedError (strategy="auto")

Collection.get(id: str, include_vector: bool = False) -> SearchResult
    Raises: VectorNotFoundError

Collection.update(id: str, vector=None, metadata: dict = None) -> None
    Raises: VectorNotFoundError

Collection.delete(id: str) -> None
    Raises: VectorNotFoundError

Collection.count() -> int

Collection.config -> CollectionConfig  (property)

Collection.estimate_memory(n_vectors: int) -> MemoryEstimate