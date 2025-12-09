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

from nahual.utils import dsatur


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


def process_data(
    pixels: np.ndarray,
    address: str,
    session_id: str,
    channel_to_segment:int,
    input_dimorder: str = "NZYX",
    extra_args:tuple[tuple[str,tuple[str,str]]]=(("refine_outlines", ("", "true")), ("with_edgemasks", ("", "true")), ("with_masks", ("", "true"))),
) -> list[dict[str, np.ndarray]]:
    """Sends image data to a baby-phone server session for segmentation.

    This function sends a multipart-encoded POST request to the `/segment`
    endpoint of the baby-phone server to initiate processing. It then sends a
    GET request to the same endpoint to retrieve the results. The POST
    request includes the image data, its dimensions, bit depth, and any extra
    processing arguments.

    Parameters
    ----------
    pixels : numpy.ndarray
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
    `pixels.tobytes(order="F")`. The bit depth is hardcoded to "8".
    The initial parts of the multipart POST request (`dims`, `bitdepth`,
    `pixels`) must be in a fixed order.
    """
    pixels = pixels[:, channel_to_segment]
    
    # Convert to uint8
    # TODO check if BABY supports uint16
    if pixels.dtype == np.uint16:
        pixels = ((pixels / 65536) * 256).astype(np.uint8)

    # Convert from the input format to NYXZ
    reordered = reorder_dims(pixels, input_dimorder=input_dimorder, output_dimorder="NYXZ")
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

    edgemask_labels = [
        {k: out_pertile[k] for k in ("edgemasks", "cell_label", "masks")}
        for out_pertile in outputs
    ]

    pertile_nyx = []
    pertile_layers = []
    
    for tile in edgemask_labels:
        labels = tile["cell_label"]
        
        nyx = np.zeros((0, *pixels.shape[1:3]), dtype=int)
        if len(labels): # Cover case of tiles 
            edgemasks = tile["edgemasks"]
            masks = tile["masks"]
            xs_max = [max(m[0]) for m in edgemasks]
            ys_max = [max(m[1]) for m in edgemasks]
            xs_min = [min(m[0]) for m in edgemasks]
            ys_min = [min(m[1]) for m in edgemasks]
            assert all([x <= pixels.shape[-2] for x in xs_max])
            assert all([y <= pixels.shape[-1] for y in ys_max])
            assert all([x > 0 for x in xs_min])
            assert all([y > 0 for y in ys_min])

            mapper_label_layer = get_layers_from_edgemasks(edgemasks)
            pertile_layers.append(mapper_label_layer)

            n_layers = max(mapper_label_layer.values()) + 1
            nyx = np.zeros((n_layers, *pixels.shape[-2:]), dtype=int)

            # Place the cells back in their corresponding layer
            # [[max(m) for m in mask_set] for mask_set in masks]
            for object_index, layer in mapper_label_layer.items():
                x,y = np.array(masks[object_index])-1
                nyx[layer,x-1,y-1] = labels[object_index]
            
        pertile_nyx.append(nyx)

    
    return pertile_nyx


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

def generate_adjacency_from_overlapping(overlapping_indices) -> list[tuple[int,int]]:
    """
    The input is the object labels for overlapping indices.
    The output is a tuple where the first value are the unique indices (from the input) and the output is an adjacency list representation (lists with all the connected nodes.).
    """
    unique_indices = sorted(set([y for x in overlapping_indices for y in x]))
    n_nodes = len(unique_indices)
    adjacency_graph = [[] for _ in  range(n_nodes)]
    for (node1, node2) in overlapping_indices:
        adjacency_graph[unique_indices.index(node1)].append(unique_indices.index(node2))
        adjacency_graph[unique_indices.index(node2)].append(unique_indices.index(node1))
        
    return unique_indices, adjacency_graph
    
def get_layers_from_edgemasks(edgemasks:list[np.ndarray])->list[int,int]:
    """Produces a dictionary that determines the how to distribute labels across multiple stacks so
    objects do not overlap. It outputs a dictionary with the original label and its corresponding layer.

    NOTE: This uses the DSatur algorithm, which will not scales well as overlaps increase in number.
    """
    if not len(edgemasks): # Cover empty case
        return {}
    
    overlapping_indices = overlap_from_edgemasks(edgemasks)

    # Convert list of overlapping indices into adjacency graph
    unique_indices, adjacency_graph = generate_adjacency_from_overlapping(overlapping_indices)

    # Case where there are no overlapping
    layers_d = {object_label:0 for object_label in range(len(edgemasks))}
    if not len(adjacency_graph):
        return layers_d
    
    # Use a colouring algorithm to find the smallest number of stacks needed to represent all cells
    layers = dsatur(adjacency_graph)

    n_layers = max(len(layers), 1)
    # Place all objects on the bottom layer
    # Overwrite the overlapping ones
    for i,layer in enumerate(layers):
        layers_d[unique_indices[i]] = layer

    return layers_d


def get_edgemasks_case(kind) -> np.ndarray:
    # tiny 6×6 label image
    
    empty = np.array([], dtype=int)
    zeros = np.zeros((6,6), dtype=int)
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

    overlap_edgemasks = matrix_to_edgemasks(overlap)
    # Add overlaps here
    overlap_edgemasks[1] = np.concatenate((overlap_edgemasks[1], ((3, 4), (3, 3))), axis=1)

    examples = {
        "empty": empty,
        "zeros": zeros,
        "no_overlap": non_overlap,
    }
    examples = {k:matrix_to_edgemasks(v) for k,v in examples.items()}
    examples["overlap"] =  overlap_edgemasks
    return examples[kind]


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

# Now we need to cover for empty cases to count the number of layers in the new dimension:
# Let N be the number of overlapping layers
# Overlapping and non-overlapping: N
# Overlapping only: N
# Non-overlapping only: 1
# No objects: Return empty array with dimensions (NZYX)

# if __file__ == "main":
# address = "http://0.0.0.0:5101"  # URL to reach baby-phone
# modelset = "yeast-alcatras-brightfield-sCMOS-60x-1z"
# result = run_sample(address, modelset)

# %%
# for case_ in ("overlap", "no_overlap", "zeros", "empty"):
#     if case_=="zeros":
#         breakpoint()
#     example = get_edgemasks_case(case_)
#     layers = get_layers_from_edgemasks(example)
#     print(f"{case_}: {layers}")

# %% Visualisation help
# max_size = max([np.array(x).max() for x in example])
# masks = np.zeros((max_size+1,max_size+1), dtype=int)
# for object_id, (xcoords, ycoords) in enumerate(example, 1):
#     for x,y in zip(xcoords, ycoords):
        # masks[x,y] = object_id

