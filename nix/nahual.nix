{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  # setuptools,
  numpy,
  hatchling,
  requests,
  pynng,
}:
buildPythonPackage {
  pname = "nahual";
  version = "0.0.1-unstable-2025-08-09";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "afermg";
    repo = "nahual";
    rev = "";
    sha256 = "";
  };

  nativeBuildInputs = [
  ];

  build-system = [
    hatchling
  ];

  dependencies = [
    numpy
    pynng
    requests
  ];

  pythonImportsCheck = [
    "pynng"
  ];

  meta = {
    description = "Python bindings for Nanomsg Next Generation";
    homepage = "https://github.com/codypiersall/pynng";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ afermg ];
    platforms = lib.platforms.all;
  };
}
