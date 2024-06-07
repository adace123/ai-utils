{
  description = "Various LLM utility scripts managed through poetry/poetry2nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    poetry2nix,
    ...
  }: let
    systems = ["x86_64-linux" "aarch64-darwin"];
    forEachSystem = nixpkgs.lib.genAttrs systems;
    forEachPkgs = f: forEachSystem (system: f (import nixpkgs {inherit system;}));
  in {
    packages = forEachPkgs (pkgs: let
      inherit (poetry2nix.lib.mkPoetry2Nix {inherit pkgs;}) mkPoetryApplication defaultPoetryOverrides;
    in {
      default = mkPoetryApplication {
        projectDir = ./.;
        python = pkgs.python311;
        preferWheels = true;
        overrides = defaultPoetryOverrides.extend (final: prev: rec {
          pyee = prev.pyee.overridePythonAttrs (old: {
            postPatch = "";
          });
          setuptools = prev.setuptools.overridePythonAttrs (old: rec {
            version = "70.0.0";
            src = pkgs.fetchFromGitHub {
              owner = "pypa";
              repo = "setuptools";
              rev = "refs/tags/v${version}";
              hash = "sha256-LXF3R9zfWylvihP2M8N94/IlgifwxUwKJFhtvcXEPB0=";
            };
          });
        });
      };
    });
  };
}
