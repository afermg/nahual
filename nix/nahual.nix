{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  # setuptools,
  numpy,
}:
buildPythonPackage {
  pname = "pynng";
  version = "0.0.1-unstable-2025-08-09";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "afermg";
    repo = "nahual";
    rev = "2179328f8a858bbb3e177f66ac132bde4a5aa859";
    sha256 = "";
  };

  nativeBuildInputs = [
  ];

  build-system = [
    # setuptools
  ];

  dependencies = [
    numpy
    # pynng
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
