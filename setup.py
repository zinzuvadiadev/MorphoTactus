from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'MorphoTactus'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Message files
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='zinzuvadiadev08@gmail.com',
    description='ROS-based hybrid defect detection system for mechanical parts using OpenCV contouring and SIFT feature matching',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_source_node = MorphoTactus.image_source_node:main',
            'preprocessing_node = MorphoTactus.preprocessing_node:main',
            'hybrid_detector_node = MorphoTactus.hybrid_detector_node:main',
            'video_viewer_node = MorphoTactus.video_viewer_node:main',
            'simple_video_viewer_node = MorphoTactus.simple_video_viewer:main',
            'realtime_video_viewer_node = MorphoTactus.realtime_video_viewer:main',
        ],
    },
)
