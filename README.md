# qjulia2
## Overview
This is a ray-tracing program specialized in rendering quaternion julia set,
completely written in native C++ from the scratch.

The ray-tracing pipeline is inspired heavily from [*PBRT*](https://www.pbrt.org/),
while the distance estimation algorithm was learnt from [1] and [2].
The output is a [PPM](http://netpbm.sourceforge.net/doc/ppm.html) image.
## Install and Environment

This project is developed and tested on Ubuntu 16.04 with GCC 5.4,
built with [Cmake](https://cmake.org/).

* Install build tools
```bash
$ sudo apt install build-essential cmake git
```
* Clone the repository and initialize the submodules
```bash
$ git clone --recursive https://github.com/zhehangd/qjulia2.git
```
* Compile the program
```bash
$ cd qjulia2
$ mkdir build && cd build
$ cmake ..
$ make -j4
```

## Usage

```bash
./src/qjulia -i ../data/example_scene.txt
```
There are a few options can be used:
* `-i <file>, --scene_file <file>`

  Specifies a scene description file. It describes what content to render.

* `-t <n>, --num_threads <n>`

  Specifies the number of threads used in rendering.

* `-s <w>x<h>, --size <w>x<h>`

  Specifies the generated image size, i.e. "-s 1920x1080"

* `-o <file>, --output_file <n>`

  Specifies the output file name, which should end with ".ppm".
  By default "output.ppm".

## Scene Description

A scene description file consists a set of blocks, such as
```
shape julia3d fractal_shape_1 {
  max_iterations 200
  max_magnitude 10.0
  bounding_radius 3.0
  constant -0.2,0.8,0,0
}
```
Each block describes an instance added to the cene manager.
A block begins with a line of header, in format
```
<TYPE> <IMPLEMENTATION> <NAME> {
  ...
}
```

There are at total 7 types:
  * `Object`
  * `Shape`
  * `Transform`
  * `Material`
  * `Light`
  * `Camera`
  * `Scene`

Entities of the same type must use unique names.

TBD


## Issues and Limitations

* Although the change of object position should be handled by
the `Transform` component, many shapes implement their own position
attribute. This may give some wired result if both set.

* The distance estimation method is used to find the surface of
fractals. Though every fast, the surface is too smooth and many
appealing details are lost.

* Entity management is not complete.

## Reference
[1] Hart, John C., Daniel J. Sandin, and Louis H. Kauffman. "Ray tracing deterministic 3-D fractals." ACM SIGGRAPH Computer Graphics. Vol. 23. No. 3. ACM, 1989.
[2] https://www.cs.cmu.edu/~kmcrane/Projects/QuaternionJulia/paper.pdf