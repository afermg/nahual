"""
This client matches baby-phone's API (https://github.com/afermg/baby). Run baby phone using `baby-phone` on the CLI.

This client differs from the other ones, as BABY uses an HTTP server method.

Example:
address = "http://0.0.0.0:5101"  # URL to reach baby-phone
modelset = "yeast-alcatras-brightfield-sCMOS-60x-1z"
result = run_sample(address, modelset)


"""

import numpy as np
import requests


def list_sessions(address: str):
    """List running sessions on the baby-phone server.

    Sends a GET request to the /sessions endpoint to retrieve a list of
    currently active processing sessions.

    Parameters
    ----------
    address : str
        The URL of the baby-phone server (e.g., 'http://0.0.0.0:5101').

    Returns
    -------
    list or str
        A list of dictionaries, where each dictionary represents a running
        session, if the request is successful. Otherwise, returns the error
        text from the server response.

    """
    r = requests.get(address + "/sessions")
    running_sessions = r.json() if r.ok else r.text
    return [x["id"] for x in running_sessions]


def get_server_info(address: str):
    """Get information about the baby-phone server.

    This function queries the server for its version, the list of available
    models with metadata, and the currently running sessions.

    Parameters
    ----------
    address : str
        The URL of the baby-phone server.

    Returns
    -------
    dict
        A dictionary containing server information with the following keys:
        'version' : str
            The server version.
        'models' : list
            A list of available models on the server.
        'running_sessions' : list
            A list of currently active sessions.

    """
    r = requests.get(address)
    baby_version = r.json() if r.ok else r.text

    r = requests.get(address + "/models?meta=true")
    available_models = r.json() if r.ok else r.text

    return {
        "version": baby_version,
        "models": available_models,
        "running_sessions": list_sessions(address),
    }


def load_model(address: str, modelset: str):
    """Load a model on the baby-phone server and create a session.

    This function first queries the available models and then sends a request
    to the baby-phone server to load a specified model set. This creates a
    new processing session and returns its unique identifier.

    Parameters
    ----------
    address : str
        The URL of the baby-phone server.
    modelset : str
        The name of the model set to load.

    Returns
    -------
    str
        The unique identifier for the newly created session.

    Raises
    ------
    Exception
        If the server fails to load the model and create a session.

    """
    r = requests.get(address + "/models")
    r.json() if r.ok else r.text
    r = requests.get(f"{address}/session/{modelset}")
    if not r.ok:
        raise Exception(f"{r.status_code}: {r.text}")
    session_id = r.json()["sessionid"]

    return session_id


def run_sample(
    address: str,
    modelset: str = "yeast-alcatras-brightfield-sCMOS-60x-5z",
    seed: int = 42,
):
    """Generate and send a sample image for processing.

    Creates a random sample image, ensures a processing session is active on the
    baby-phone server, sends the image for segmentation, and returns the result.
    If no sessions are running, it will create one using the specified model set.

    Parameters
    ----------
    address : str
        The URL of the baby-phone server.
    modelset : str, optional
        The name of the model set to load if a new session needs to be created.
        Defaults to "yeast-alcatras-brightfield-sCMOS-60x-1z".
    seed : int, optional
        Seed for the random number generator to create the sample image.
        By default, 42.

    Returns
    -------
    dict or str
        A dictionary containing the segmentation results from the server if the
        request is successful. Otherwise, returns the error text from the
        server response.

    See Also
    --------
    list_sessions : List running sessions on the server.
    load_model : Load a model and create a new session.
    process_data : Send data for processing within an existing session.

    Notes
    -----
    This function internally generates a random image of shape (2, 80, 120, 1)
    and dtype 'uint8' for processing. The image generation is seeded by the
    `seed` parameter for reproducibility.

    """
    rng = np.random.default_rng(seed)

    running_sessions = list_sessions(address)
    session_id = (
        running_sessions[0]
        if len(running_sessions) > 0
        else load_model(address, modelset)
    )

    # Create suitable N x H x W x Z array
    # dtype must be either uint8 or uint16
    img = rng.integers(2**8, size=(2, 80, 120, 1), dtype="uint8")

    output = process_data(img, address, session_id, input_dimorder="NYXZ")

    return output


def process_data(
    img: np.ndarray,
    address: str,
    session_id: str,
    input_dimorder: str = "NZYX",
    extra_args=(("refine_outlines", ("", "true")), ("with_edgemasks", ("", "true"))),
) -> list[dict[str, np.ndarray]]:
    """Sends image data to a baby-phone server session for segmentation.

    This function sends a multipart-encoded POST request to the `/segment`
    endpoint of the baby-phone server to initiate processing. It then sends a
    GET request to the same endpoint to retrieve the results. The POST
    request includes the image data, its dimensions, bit depth, and any extra
    processing arguments.

    Parameters
    ----------
    img : numpy.ndarray
        The image data to be processed. Expected to be a 4D NumPy array with
        shape (N, H, W, Z), where N is the number of images, H is height, W
        is width, and Z is the number of z-slices.
    address : str
        The URL of the baby-phone server (e.g., 'http://0.0.0.0:5101').
    session_id : str
        The unique identifier for the processing session on the server.
    extra_args : tuple, optional
        A tuple of extra arguments to be passed in the multipart request.
        Each element is a tuple that sets a keyword argument for
        `BabyCrawler.step`. Defaults to enabling outline refinement and
        including edge masks in the output.

    Returns
    -------
    list of dict
        A list of dictionaries containing the segmentation results. Each
        dictionary corresponds to an image in the input batch and contains
        the 'edgemasks' and 'cell_label' keys from the server response.

    Raises
    ------
    Exception
        If the GET request to retrieve segmentation results from the server
        fails or returns a non-OK status code.

    Notes
    -----
    The function operates in two stages:
    1. A POST request to submit the data for processing.
    2. A GET request to fetch the completed results.

    The image data is serialized into bytes using Fortran order via
    `img.tobytes(order="F")`. The bit depth is hardcoded to "8".
    The initial parts of the multipart POST request (`dims`, `bitdepth`,
    `img`) must be in a fixed order.
    """
    # Convert to uint8
    # TODO check if BABY supports uint16
    if img.dtype == np.uint16:
        img = ((img / 65536) * 256).astype(np.uint8)

    # Convert from the input format to NYXZ
    reordered = reorder_dims(img, input_dimorder=input_dimorder, output_dimorder="NYXZ")
    # Initiate a multipart-encoded request
    requests.post(
        f"{address}/segment?sessionid={session_id}",
        files=[
            # The ordering of these parts must be fixed
            ("dims", ("", str(list(reordered.shape)))),
            ("bitdepth", ("", "8")),
            ("img", ("", reordered.tobytes(order="F"))),
            # Optionally specify additional parts that set
            # any kwargs accepted by BabyCrawler.step (ordering
            # is no longer fixed)
            *extra_args,
        ],
    )

    # Request results
    r = requests.get(f"{address}/segment?sessionid={session_id}")
    if not r.ok:
        raise Exception(f"{r.status_code}: {r.text}")
    outputs = r.json()

    edgemasks_labels = [
        {k: out_pertile[k] for k in ("edgemasks", "cell_label")}
        for out_pertile in outputs
    ]

    # TODO convert edgemasks to flat 2d masks
    # Return using updated labels?
    return edgemasks_labels


def reorder_dims(
    img: np.ndarray, input_dimorder: str, output_dimorder: str
) -> np.ndarray:
    """Reorders the dimensions of an array based on string specifications.

    Parameters
    ----------
    img : np.ndarray
        The input array.
    input_dimorder : str
        A string representing the order of dimensions in the input array,
        e.g., 'NZYX' for Height, Width, Channels. Each character represents
        one dimension.
    output_dimorder : str
        A string representing the desired order of dimensions in the output
        array, e.g., 'NYXZ'. It must contain the same characters as
        `input_dimorder`.

    Returns
    -------
    np.ndarray
        The array with reordered dimensions.

    Raises
    ------
    ValueError
        If dimension orders are invalid or incompatible with the array shape.
    """
    assert sorted(input_dimorder) == sorted(
        output_dimorder
    ), "The strings must be permutations of each other"
    axes_permutation = [input_dimorder.index(dim) for dim in output_dimorder]

    return np.transpose(img, axes_permutation)


def matrix_to_edgemasks(arr: np.ndarray) -> list[tuple[int, int]]:
    uniq = np.unique(arr)
    uniq = uniq[uniq > 0]
    d = {}
    for k in uniq:
        d[k] = np.array(np.where(arr == k))

    return list(d.values())


def get_edgemasks_example(kind) -> np.ndarray:
    # tiny 6×6 label image
    non_overlap = np.array(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [2, 2, 2, 0, 0, 0],
            [2, 0, 2, 3, 3, 3],
            [2, 2, 2, 3, 0, 3],
            [0, 0, 0, 3, 3, 3],
        ]
    )
    overlap = np.array(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [2, 2, 2, 2, 0, 0],
            [2, 0, 0, 3, 3, 3],
            [2, 2, 2, 3, 0, 3],
            [0, 0, 0, 3, 3, 3],
        ]
    )

    edgemasks = matrix_to_edgemasks(overlap)
    # Add overlaps here
    edgemasks[1] = np.concatenate((edgemasks[1], ((3, 4), (3, 3))), axis=1)

    examples = {
        "overlap": edgemasks,
        "non_overlap": matrix_to_edgemasks(non_overlap),
    }
    return examples[kind]


def overlap_from_edgemasks(edgemasks: list[np.ndarray]) -> list[tuple[int, int]]:
    edges = np.asarray(
        [(np.min(edge_set, axis=1), np.max(edge_set, axis=1)) for edge_set in edgemasks]
    )
    # Masking can probably occur with a matrix of a specific shape
    # For now I will do iteration
    overlaps = []
    for i, ((left, top), (right, bottom)) in enumerate(edges):
        for j, ((other_left, other_top), (other_right, other_bottom)) in enumerate(
            edges
        ):
            if i >= j:
                continue
            # print(f"{left} >= {other_left} and {other_right} <= {right}")
            # print(f"{top} >= {other_top} and {bottom} <= {other_bottom}")
            if (right >= other_left and left <= other_right) and (
                bottom >= other_top and top <= other_bottom
            ):
                overlaps += [(i, j)]

    return overlaps


def group_edgemasks(edgemasks: list[np.ndarray]):
    # Cells are considered overlapping if for any pixel (x_1,y_1) there is a pair of pixels (x_2,y_2), (x_3,y_3) belonging to a different mask
    # where (x_2 < x_1 < x_3) and (y_2 < y_1 < y_3).
    ijv = edgemasks_to_ijv(edgemasks)
    overlaps = overlap


def edgemasks_to_ijv(masks: list[np.ndarray]) -> np.ndarray:
    """Convert edgemasks to an ijv matrix.

    An edgemask is a list of 2-d arrays, where each list"""
    lens = [len(x[0]) for x in masks]
    ijv = np.zeros((np.sum(lens), 3), dtype=np.uint16)
    position = np.zeros(len(lens) + 1, dtype=np.uint16)
    position[1:] = np.cumsum(lens)
    for v, (x, y) in enumerate(masks):
        slc = slice(position[v], position[v + 1])
        ijv[slc, 0] = x
        ijv[slc, 1] = y
        ijv[slc, 2] = v

    return ijv


def index_isin(
    x: np.ndarray,
    y: np.ndarray,
    i_dtype={"names": ["i", "j", "v"], "formats": [np.uint16, np.uint16, np.uint16]},
) -> np.ndarray:
    """
    Find those elements of x that are in y.

    Both arrays must be arrays of integer indices,
    such as (trap_id, cell_id).

    NOTE: This is (AFAIK) the fastest way to do this.
    This is more of a database-querying task but
    it is very common that we ought to do it with numpy.
    """
    x = np.ascontiguousarray(x, dtype=np.int16)
    y = np.ascontiguousarray(y, dtype=np.int16)
    xv = x.view(i_dtype)
    inboth = np.intersect1d(xv, y.view(i_dtype))
    x_bool = np.isin(xv, inboth)
    return x_bool

# %%
example = get_edgemasks_example("overlap")
overlapping_indices = overlap_from_edgemasks(example)

# Visualisation help
max_size = max([np.array(x).max() for x in example])
masks = np.zeros((max_size+1,max_size+1), dtype=int)
for object_id, (xcoords, ycoords) in enumerate(example, 1):
    for x,y in zip(xcoords, ycoords):
        masks[x,y] = object_id

def colour_object_labels(graph_edge_representation:list[tuple[int,int]]):
    unique_indices, adjacency_list = generate_adjacency_from_overlapping(graph_edge_representation)

    from nahual.utils import dsatur

def generate_adjacency_from_overlapping(overlapping_indices) -> list[tuple[int,int]]:
    """
    The input is the actual cell ids with overlapping indices.
    The output is a tuple where the first value are the unique indices (from the input) and the output is an adjacency list representation (lists with all the connected nodes.).
    """
    unique_indices = sorted(set([y for x in overlapping_indices for y in x]))
    n_nodes = len(unique_indices)
    adjacency_graph = [[] for _ in  range(n_nodes)]
    for (node1, node2) in overlapping_indices:
        adjacency_graph[unique_indices.index(node1)].append(node2)
        adjacency_graph[unique_indices.index(node2)].append(node1)
        
    return unique_indices, adjacency_graph
    

tmp = generate_adjacency_from_overlapping(overlapping_indices)
