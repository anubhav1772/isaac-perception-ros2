from setuptools import find_packages, setup

package_name = 'isaac_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anubhav1772',
    maintainer_email='anubhavsingh55182@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'camera_node = isaac_bridge.camera_node:CameraPublisher',
        'zmq_bridge = isaac_bridge.zmq_bridge_node:main',
        'cmd_vel_zmq_bridge = isaac_bridge.cmd_vel_zmq_bridge:main',
        ],
    },
)
