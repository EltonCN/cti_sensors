from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'cti_sensors'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eltsu',
    maintainer_email='43186596+EltonCN@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'er_ukf_imu = cti_sensors.er_ukf_imu:main',
            'dataset_test= cti_sensors.dataset_test:main',
        ],
    },
)
