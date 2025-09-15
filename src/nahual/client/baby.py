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


import numpy as np


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


def get_data():
    mask = np.array([[0, 1, 1, 1, 2, 2], [1, 0, 2, 3, 3, 1]])

    return (mask, mask + 1)


def find_overlap_from_ijv(ijv: np.ndarray, uniq, col: int) -> list[np.ndarray]:
    """
    Find ijv subsets with at least one value
    "sandwiched" between two over `col`.
    """
    v_col = 2
    argsort = ijv[:, col].argsort()
    sorted_labels = ijv[argsort, v_col]

    # Find the boundaries using a boolean array per label
    vals = uniq[:, None] == sorted_labels
    left = vals.argmax(axis=1)
    right = vals[:, ::-1].argmax(axis=1)

    # These "inbetween" pixels may not be homogeneous,
    # that is why I am using lists
    between = [ijv[argsort][start:end] for start, end in zip(left, (len(ijv) - right))]

    arg_overlap = [np.where(x[:, v_col] != x_val)[0] for x, x_val in zip(between, uniq)]
    overlap_ijv = [between[i][x] for i, x in enumerate(arg_overlap)]

    return overlap_ijv


def find_overlapping_pixels(ijv: np.ndarray) -> dict[int, np.ndarray]:
    uniq = np.unique(ijv[:, v_col])
    x_overlap = find_overlap_from_ijv(ijv, col=0, uniq=uniq)
    y_overlap = find_overlap_from_ijv(ijv, col=1, uniq=uniq)

    result =  {label: x[index_isin(x, y)[:,0]] for label, x, y in zip(uniq, x_overlap, y_overlap)}
    return result


def group_edgemasks(masks: list[np.ndarray]):
    # Cells are considered overlapping if for any pixel (x_1,y_1) there is a pair of pixels (x_2,y_2), (x_3,y_3) belonging to a different mask
    # where (x_2 < x_1 < x_3) and (y_2 < y_1 < y_3).
    ijv = edgemasks_to_ijv(masks)


def edgemasks_to_ijv(masks: list[np.ndarray]) -> np.ndarray:
    """Convert edgemasks to an ijv matrix"""
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


# def edge_to_ijv():
# def edge_to_nzyx():
# """Convert a representation of edgemasks into a dense representation of overlapping
# masks where the first dimension ensures no overlap.
# """
# 1.Check any overlap
# 1.1 Min-on-max
# 2. If exists, identify which sets of masks overlap
# 3. Solve as a linear programming problem
# 3.1 Find the smallest number of subgroups
# 3.2 Sort by size
# 3.3 Assign subgroup to all non-overlapping masks
# 3.4


def index_isin(x: np.ndarray, y: np.ndarray, i_dtype = {"names": ["i", "j", "v"], "formats": [np.uint16, np.uint16, np.uint16]}) -> np.ndarray:
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


# Old code
# There may be a way to do this with broadcasting
# but it eluded me
# simple = np.array([0,2,1,3])
# ijv = np.stack((simple,simple, (0,0,1,1))).T

# vals = np.unique(ijv[:,2])
# occurrences = vals[:,None] == ijv[:,2]

# less_x = np.less_equal.outer(ijv[:,0],ijv[:,0])
# greater_x = np.greater_equal.outer(ijv[:,0],ijv[:,0])
# less_y = np.less_equal.outer(ijv[:,1],ijv[:,1])
# greater_y = np.greater_equal.outer(ijv[:,1],ijv[:,1])

# Places where the pixels sits between two others
# TODO Fix: ensure the output corresponds to in-between
# x = i, y = lower than i?, z = higher than i?
# inbetween_x = (less_x[..., None] & greater_x[None])
# correct = np.array([less_x[i][:,None] & greater_x[i] for i in range(len(less_x))])
# inbetween_x = (less_x & greater_x[:,None]).transpose((2,0,1))
# inbetween_y = (less_y & greater_y[:,None]).transpose((2,0,1))
# Remove the ones from the same object (equal_v)
# Remove the ones where the pixels belong to the same object
# array([[,
#         [ True, False, False, False],
#         [ True, False, False, False],
#         [ True, False, False, False]],

#        [[False, False, False, False],
#         [ True,  True, False, False],
#         [ True,  True, False, False],
#         [ True,  True, False, False]],

#        [[False, False, False, False],
#         [False, False, False, False],
#         [ True,  True,  True, False],
#         [ True,  True,  True, False]],

#        [[False, False, False, False],
#         [False, False, False, False],
#         [False, False, False, False],
#         [ True,  True,  True,  True]]])

# inbetween_x = less_x & greater_x[:,None]
# inbetween_y = less_y & greater_y[:,None]
# Inbetween = inbetween_x & inbetween_y

# # Filter where it is each object
# overlaps = np.where(inbetween & equal_v)

# return np.unique(ijv[overlalps,2])
