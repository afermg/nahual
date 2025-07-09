{
  lib,
  pkgs,
  python3Packages,
}:
let
  callPackage = lib.callPackageWith (pkgs // packages // python3Packages);
  packages = {
    nahual = callPackage ./nahual.nix { };
  };
in
packages
