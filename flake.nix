{
  description = "Development environment for Ngo Tan Dat's Portfolio (Astro + TypeScript)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
    in
    {
      devShells = forEachSystem (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            nodejs_22
            pnpm
          ];

          shellHook = ''
            echo "========================================================"
            echo " 🚀 Data Engineer Portfolio Environment (Astro + TS)"
            echo " Node: $(node --version) | PNPM: $(pnpm --version)"
            echo " Commands: 'pnpm dev' | 'pnpm build'"
            echo "========================================================"
          '';
        };
      });
    };
}
