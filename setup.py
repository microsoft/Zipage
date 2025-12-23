from setuptools import setup, find_packages


zipage_packages = find_packages(include=["zipage", "zipage.*"])
nanovllm_packages = find_packages(where="third_party", include=["nanovllm", "nanovllm.*"])
all_packages = zipage_packages + nanovllm_packages 
package_dir = {}

for pkg in nanovllm_packages:
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
    ]
)
