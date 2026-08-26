from setuptools import find_packages, setup
from glob import glob

package_name = 'ah_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('models/*.onnx')),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ferestrada',
    maintainer_email='fernaest213@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ah_policy_node = ah_driver.ah_policy_node:main',
        ],
    },
)
