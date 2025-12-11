from setuptools import setup, find_packages


# Find zipage packages  
zipage_packages = find_packages(include=["zipage", "zipage.*"])

# Find nanovllm packages from third_party
nanovllm_packages = find_packages(where="third_party", include=["nanovllm", "nanovllm.*"])

# Combine all packages
all_packages = zipage_packages + nanovllm_packages

# Create package_dir mapping for nanovllm packages
package_dir = {}
for pkg in nanovllm_packages:
    # Map nanovllm.xxx to third_party/nanovllm/xxx
    package_dir[pkg] = f"third_party/{pkg.replace('.', '/')}"

setup(
    name="zipage",
    version="0.1.0",
    description="Zipage: Maintain High Request Concurrency for LLM Reasoning through Compressed PagedAttention",
    packages=all_packages,
    package_dir=package_dir,
    package_data={
        "": ["*.so", "*.pyi", "*.typed"],
    },
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0",
        "transformers>=4.51.3",
        "triton>=3.2.0",
    ],
    extras_require={
        "examples": [
            "datasets",
        ],
    },
)
