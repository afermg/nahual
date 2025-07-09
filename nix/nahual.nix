{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  hatchling,
  numpy,
  pynng,
  requests,
}:
buildPythonPackage {
  pname = "nahual";
  version = "0.0.1-unstable-2025-08-09";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "afermg";
    repo = "nahual";
    rev = "302a00c61b989f20553681d0edc3130e8691e222";
    sha256 = "sha256-RTi05oUB4QmFRw9ZgUseelAN//lhHTJWTNUqPocNa4M=sha256-RTi05oUB4QmFRw9ZgUseelAN//lhHTJWTNUqPocNa4M=";
  };

  build-system = [
    hatchling
  ];

  dependencies = [
    numpy
    pynng
    requests
  ];

  pythonImportsCheck = [
    "nahual"
  ];

  meta = {
    description = "Python bindings for Nanomsg Next Generation";
    homepage = "https://github.com/codypiersall/pynng";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ afermg ];
    platforms = lib.platforms.all;
  };
}
