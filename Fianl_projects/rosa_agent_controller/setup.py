from setuptools import find_packages, setup

package_name = 'rosa_agent_controller'

setup(
    name=package_name,
    version='0.0.0',  # 修改为与package.xml相同的版本号
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'requests'],
    zip_safe=True,
    maintainer='biodyn',
    maintainer_email='biodyn@todo.todo',
    description='TODO: Package description',  # 保持与package.xml一致
    license='TODO: License declaration',  # 保持与package.xml一致
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rosa_controller = rosa_agent_controller.rosa_controller_node:main',
        ],
    },
)
