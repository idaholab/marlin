/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedJITTensor.h"
#include "TensorBuffer.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>
#include <chrono>

#if 0
TEST(BenchmarkParsedTensors, Time)
{
  ParsedTensor F1;
  ParsedJITTensor F2;
  std::string variables = "a, b, c";

  const std::size_t N = 2000;
  auto A = MooseTensor::createBuffer({N, N}, {6.0, 6.0});
  auto B = MooseTensor::createBuffer({N, N});
  auto C = MooseTensor::createBuffer({N, N});

  auto & a = A.data();
  auto & b = B.data();
  auto & c = C.data();

  auto [x, y] = A.getAxis();

  // function
  a = x;
  b = y;
  c = 2.0 * x * y;

  F1.Parse("a * b + c + log(a+0.1) + a*a*a*a - b*b*b ", variables);
  F1.setupTensors();

  F2.Parse("a * b + c + log(a+0.1) + a*a*a*a - b*b*b ", variables);
  F2.setupTensors();

  // auto gold = a * b + c;

  if (!torch::cuda::is_available())
    mooseError("CUDA is not available! :-(");
  mooseInfo("CUDA DEVICE COUNT: ", torch::cuda::device_count());

  a.to(torch::kFloat32);
  b.to(torch::kFloat32);
  c.to(torch::kFloat32);
  a.to(torch::kCUDA);
  b.to(torch::kCUDA);
  c.to(torch::kCUDA);

  a.cuda();
  if (!a.is_cuda())
    mooseInfo("Tensor a should be on the GPU but is not");

  // time non-JIT
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  std::vector<const torch::Tensor *> params{&a, &b, &c};

  {
    if (!F1.Eval(params).is_cuda())
      mooseInfo("non-JIT result is not CUDA");
    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < 1000; i++)
      F1.Eval(params);

    auto t2 = high_resolution_clock::now();

    MooseTensor::printTensorInfo(F1.Eval(params));

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    Moose::out << "non-JIT : " << ms_double.count() << "ms\n";
  }

  {
    if (!F2.Eval(params).is_cuda())
      mooseInfo("JIT result is not CUDA");
    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < 1000; i++)
      F2.Eval(params);

    auto t2 = high_resolution_clock::now();

    MooseTensor::printTensorInfo(F2.Eval(params));

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    Moose::out << "    JIT : " << ms_double.count() << "ms\n";
  }
}
#endif
