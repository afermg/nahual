{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        libList = [
          # Add needed packages here
          pkgs.zlib # Numpy
          pkgs.stdenv.cc.cc
        ];
      in
      with pkgs;
      rec {
        packages = pkgs.callPackage ./nix { };
        devShells = {
          default =
            let
              python_with_pkgs = pkgs.python3.withPackages (pp: [
                # Add python pkgs here that you need from nix repos
                packages.nahual
              ]);
            in
            mkShell {
              NIX_LD_LIBRARY_PATH = lib.makeLibraryPath libList;
              packages = [
                python_with_pkgs
                # python3Packages.venvShellHook
                pkgs.uv
              ]
              ++ libList;
              venvDir = "./.venv";
              postVenvCreation = ''
                unset SOURCE_DATE_EPOCH
              '';
              postShellHook = ''
                unset SOURCE_DATE_EPOCH
              '';
              shellHook = ''
                export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                uv sync --all-groups
                source .venv/bin/activate
              '';
            };
        };
      }
    );
}
# runHook venvShellHook # This runs nix python
# export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
