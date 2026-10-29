/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#ifdef NEML2_ENABLED

#include "gtest/gtest.h"
#include "neml2/misc/defaults.h"
#include "neml2/neml2.h"
#include "neml2/tensors/Scalar.h"

#include <cmath>
#include <filesystem>
#include <fstream>

// Test the JohnsonCookFlowRate model
class JohnsonCookFlowRateTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    neml2::set_default_dtype(neml2::kFloat64);

    const auto input_path = std::filesystem::temp_directory_path() / "JohnsonCookFlowRateTest.i";

    // Create NEML2 input for the Johnson-Cook model
    std::ofstream input(input_path);
    input << R"(
[Models]
  [jc]
    type = JohnsonCookFlowRate
    vonmises_stress = 'forces/s'
    equivalent_plastic_strain = 'forces/ep'
    flow_rate = 'state/gamma_rate'
    use_temperature = false
    A = 99.7e6
    B = 262.8e6
    n = 0.23
    C = 0.029
    m = 0.98
    reference_strain_rate = 1.0
  []
[]
)";
    input.close();

    // Parse the input and build the model
    _model = neml2::load_model(input_path, "jc");
    std::filesystem::remove(input_path);
  }

  std::shared_ptr<neml2::Model> _model;
  const neml2::VariableName _stress_name = "forces/s";
  const neml2::VariableName _plastic_strain_name = "forces/ep";
  const neml2::VariableName _flow_rate_name = "state/gamma_rate";
};

TEST_F(JohnsonCookFlowRateTest, BelowYield)
{
  // When stress is below yield strength, flow rate should be zero
  // For Cu: A = 99.7 MPa, so below 99.7 MPa we should have zero flow

  auto s = neml2::Scalar::full(50e6); // 50 MPa stress
  auto ep = neml2::Scalar::full(0.0); // No plastic strain

  neml2::ValueMap in;
  in[_stress_name] = s;
  in[_plastic_strain_name] = ep;

  auto out = _model->value(in);
  auto gamma_rate = out.at(_flow_rate_name);

  // Flow rate should be zero (or very small) when below yield
  EXPECT_NEAR(gamma_rate.item<double>(), 0.0, 1e-10);
}

TEST_F(JohnsonCookFlowRateTest, AtYield)
{
  // When stress equals yield strength, the NEML2 Heaviside convention gives
  // half of the reference strain rate at the transition.

  const auto ep_value = 1e-10;
  const auto sigma_y = 99.7e6 + 262.8e6 * std::pow(ep_value + 1e-10, 0.23);
  auto s = neml2::Scalar::full(sigma_y); // At yield
  auto ep = neml2::Scalar::full(ep_value);

  neml2::ValueMap in;
  in[_stress_name] = s;
  in[_plastic_strain_name] = ep;

  auto out = _model->value(in);
  auto gamma_rate = out.at(_flow_rate_name);

  // At yield, the inverted JC formula gives exp(0), multiplied by H(0) = 0.5.
  EXPECT_NEAR(gamma_rate.item<double>(), 0.5, 1e-12);
}

TEST_F(JohnsonCookFlowRateTest, AboveYield)
{
  // When stress is above yield, flow rate should be positive

  // At ep = 0.1, sigma_y = A + B*ep^n = 99.7 + 262.8*0.1^0.23 = ~99.7 + 138.5 = 238.2 MPa
  auto s = neml2::Scalar::full(300e6); // 300 MPa > yield
  auto ep = neml2::Scalar::full(0.1);  // 10% plastic strain

  neml2::ValueMap in;
  in[_stress_name] = s;
  in[_plastic_strain_name] = ep;

  auto out = _model->value(in);
  auto gamma_rate = out.at(_flow_rate_name);

  // Flow rate should be positive and > reference rate
  EXPECT_GT(gamma_rate.item<double>(), 1.0);
}

TEST_F(JohnsonCookFlowRateTest, RateSensitivity)
{
  // Higher stress should give higher flow rate (rate sensitivity)

  auto ep = neml2::Scalar::full(0.05);

  // Calculate yield stress at ep=0.05
  // sigma_y = 99.7 + 262.8 * 0.05^0.23 = ~99.7 + 107.3 = 207 MPa

  auto s1 = neml2::Scalar::full(250e6); // 250 MPa
  auto s2 = neml2::Scalar::full(350e6); // 350 MPa

  neml2::ValueMap in1, in2;
  in1[_stress_name] = s1;
  in1[_plastic_strain_name] = ep;
  in2[_stress_name] = s2;
  in2[_plastic_strain_name] = ep;

  auto out1 = _model->value(in1);
  auto out2 = _model->value(in2);

  auto rate1 = out1.at(_flow_rate_name).item<double>();
  auto rate2 = out2.at(_flow_rate_name).item<double>();

  // Higher stress should give higher flow rate
  EXPECT_GT(rate2, rate1);
}

TEST_F(JohnsonCookFlowRateTest, HardeningEffect)
{
  // Higher plastic strain should require higher stress for same flow rate

  auto s = neml2::Scalar::full(300e6); // Fixed stress

  auto ep1 = neml2::Scalar::full(0.01); // 1% plastic strain
  auto ep2 = neml2::Scalar::full(0.1);  // 10% plastic strain

  neml2::ValueMap in1, in2;
  in1[_stress_name] = s;
  in1[_plastic_strain_name] = ep1;
  in2[_stress_name] = s;
  in2[_plastic_strain_name] = ep2;

  auto out1 = _model->value(in1);
  auto out2 = _model->value(in2);

  auto rate1 = out1.at(_flow_rate_name).item<double>();
  auto rate2 = out2.at(_flow_rate_name).item<double>();

  // Higher plastic strain means higher yield stress,
  // so same applied stress gives lower flow rate
  EXPECT_GT(rate1, rate2);
}

#endif // NEML2_ENABLED
