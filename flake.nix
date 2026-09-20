{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      flake-parts,
      rust-overlay,
      crane,
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

          # Crane builds dependencies as their own derivation (buildDepsOnly),
          # so a source-only change rebuilds just rp-bot — the dep artifacts
          # stay in the store across updates. This is what makes frequent
          # deploys on a slow box (e.g. aarch64 microvm) viable.
          craneLib = (crane.mkLib pkgs).overrideToolchain (_: rustToolchain);
          src = craneLib.cleanCargoSource ./.;
          cargoArtifacts = craneLib.buildDepsOnly { inherit src; };
          rp-bot = craneLib.buildPackage {
            inherit src cargoArtifacts;
          };
        in
        {
          devShells.default = pkgs.mkShell {
            name = "rust-nightly";

            packages = [
              rustToolchain
            ];
          };

          packages.rp-bot = rp-bot;
          packages.default = rp-bot;

          checks = {
            rp-clippy = craneLib.cargoClippy {
              inherit src cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- -Dwarnings";
            };
            rp-test = craneLib.cargoTest {
              inherit src cargoArtifacts;
            };
          };
        };

      flake.nixosModules.default = import ./module.nix;
    };
}
