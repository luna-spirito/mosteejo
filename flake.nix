{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      flake-parts,
      rust-overlay,
      nixpkgs,
      ...
    }:
    let rustNightly = "2026-07-15"; in
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem =
        { config, system, ... }:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              rust-overlay.overlays.default
            ];
          };

          rustToolchain = pkgs.rust-bin.nightly.${rustNightly}.default.override {
            extensions = [
              "rust-src"
              "rust-analyzer"
              "clippy"
              "rustfmt"
            ];
          };
       in
        {
          devShells.default = pkgs.mkShell {
            name = "rust-nightly";

            packages = [
              rustToolchain
            ];
          };

          packages.rp-bot = pkgs.rustPlatform.buildRustPackage {
            pname = "rp-bot";
            version = "0.1.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;
            cargo = rustToolchain;
            rustc = rustToolchain;
          };

          packages.default = config.packages.rp-bot;
        };
    };
}
