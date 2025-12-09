{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  hatchling,
  numpy,
  pynng,
  requests,
  pytest,
  loguru,
  matplotlib,
}:
buildPythonPackage {
  pname = "nahual";
  version = "0.0.5";
  format = "pyproject";

  src = ./..;
  # src = fetchFromGitHub {
  #   owner = "afermg";
  #   repo = "nahual";
  #   rev = "d9a809aa82ee5eef59fde05c0f6fca63f6b8b184";
  #   sha256 = "sha256-QXDCLComdJj/6CTKMeF7nCzAIROhc27WXMi/QxGSU24=sha256-QXDCLComdJj/6CTKMeF7nCzAIROhc27WXMi/QxGSU24=";
  # };

  build-system = [
    hatchling
  ];

  dependencies = [
    numpy
    pynng
    requests
    pytest
    loguru
    matplotlib
  ];

  pythonImportsCheck = [
    "nahual"
  ];

  meta = {
    description = " Deploy and access image and data processing models from different environments.";
    homepage = "https://github.com/afermg/nahual";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ afermg ];
    platforms = lib.platforms.all;
  };
}
