import sys

required_version = (3, 8)

if sys.version_info[:2] < required_version:
    msg = "%s requires Python %d.%d+" % (__package__, *required_version)
    raise RuntimeError(msg)

del required_version
del sys

import pathlib
import charonload

PROJECT_ROOT_DIRECTORY = pathlib.Path(__file__).parents[3]

VSCODE_STUBS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "typings"


charonload.module_config["_c_riftcast"] = charonload.Config(
    PROJECT_ROOT_DIRECTORY,
    stubs_directory=VSCODE_STUBS_DIRECTORY,
    verbose=False,
    cmake_options={
        "ATCG_CUDA_BACKEND": "On",
        "ATCG_PYTHON_BINDINGS": "On",
        "RIFTCAST_LIB_BUILD_BINDINGS": "On",
    },
)

# import _c_torchhull  # noqa: F401
from _c_riftcast import (
    DatasetImporter,
    ATCGDatasetImporter,
    VCIDatasetImporter,
    VCIRealDatasetImporter,
    VCIRealTCPDatasetImporter,
    PanopticDatasetImporter,
    TCPDatasetImporter,
    DatasetHeader,
    DatasetType,
    IO,
    protocol,
)
