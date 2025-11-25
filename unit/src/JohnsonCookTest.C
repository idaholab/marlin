/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#ifdef NEML2_ENABLED

#include "gtest/gtest.h"
#include "neml2/base/Factory.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/misc/parser_utils.h"

// Test the JohnsonCookFlowRate model
class JohnsonCookFlowRateTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Create NEML2 input for the Johnson-Cook model
    std::string input = R"(
[Models]
  [jc]
    type = JohnsonCookFlowRate
    vonmises_stress = 'forces/s'
    equivalent_plastic_strain = 'forces/ep'
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

    // Parse the input and build the model
    auto factory = neml2::load_input(input, "");
    _model = factory->create<neml2::Model>("jc");
  }

  std::shared_ptr<neml2::Model> _model;
};

TEST_F(JohnsonCookFlowRateTest, BelowYield)
{
  // When stress is below yield strength, flow rate should be zero
  // For Cu: A = 99.7 MPa, so below 99.7 MPa we should have zero flow

  auto s = neml2::Scalar::full(50e6, _model->options());   // 50 MPa stress
  auto ep = neml2::Scalar::full(0.0, _model->options());   // No plastic strain

  neml2::ValueMap in;
  in["forces/s"] = s;
  in["forces/ep"] = ep;

  auto out = _model->value(in);
  auto gamma_rate = out.at("state/internal/gamma_rate");

  // Flow rate should be zero (or very small) when below yield
  EXPECT_NEAR(gamma_rate.item<double>(), 0.0, 1e-10);
}

TEST_F(JohnsonCookFlowRateTest, AtYield)
{
  // When stress equals yield strength at zero plastic strain,
  // flow rate should equal the reference strain rate

  // sigma_y = A + B*ep^n = A = 99.7 MPa (at ep=0)
  auto s = neml2::Scalar::full(99.7e6, _model->options());  // At yield
  auto ep = neml2::Scalar::full(1e-10, _model->options());  // Small plastic strain

  neml2::ValueMap in;
  in["forces/s"] = s;
  in["forces/ep"] = ep;

  auto out = _model->value(in);
  auto gamma_rate = out.at("state/internal/gamma_rate");

  // At yield, the inverted JC formula gives:
  // gamma_rate = eps0 * exp((1 - 1)/C) = eps0 * exp(0) = eps0 = 1.0
  EXPECT_NEAR(gamma_rate.item<double>(), 1.0, 0.1);  // Should be ~1.0
}

TEST_F(JohnsonCookFlowRateTest, AboveYield)
{
  // When stress is above yield, flow rate should be positive

  // At ep = 0.1, sigma_y = A + B*ep^n = 99.7 + 262.8*0.1^0.23 = ~99.7 + 138.5 = 238.2 MPa
  auto s = neml2::Scalar::full(300e6, _model->options());   // 300 MPa > yield
  auto ep = neml2::Scalar::full(0.1, _model->options());    // 10% plastic strain

  neml2::ValueMap in;
  in["forces/s"] = s;
  in["forces/ep"] = ep;

  auto out = _model->value(in);
  auto gamma_rate = out.at("state/internal/gamma_rate");

  // Flow rate should be positive and > reference rate
  EXPECT_GT(gamma_rate.item<double>(), 1.0);
}

TEST_F(JohnsonCookFlowRateTest, RateSensitivity)
{
  // Higher stress should give higher flow rate (rate sensitivity)

  auto ep = neml2::Scalar::full(0.05, _model->options());

  // Calculate yield stress at ep=0.05
  // sigma_y = 99.7 + 262.8 * 0.05^0.23 = ~99.7 + 107.3 = 207 MPa

  auto s1 = neml2::Scalar::full(250e6, _model->options());  // 250 MPa
  auto s2 = neml2::Scalar::full(350e6, _model->options());  // 350 MPa

  neml2::ValueMap in1, in2;
  in1["forces/s"] = s1;
  in1["forces/ep"] = ep;
  in2["forces/s"] = s2;
  in2["forces/ep"] = ep;

  auto out1 = _model->value(in1);
  auto out2 = _model->value(in2);

  auto rate1 = out1.at("state/internal/gamma_rate").item<double>();
  auto rate2 = out2.at("state/internal/gamma_rate").item<double>();

  // Higher stress should give higher flow rate
  EXPECT_GT(rate2, rate1);
}

TEST_F(JohnsonCookFlowRateTest, HardeningEffect)
{
  // Higher plastic strain should require higher stress for same flow rate

  auto s = neml2::Scalar::full(300e6, _model->options());  // Fixed stress

  auto ep1 = neml2::Scalar::full(0.01, _model->options());  // 1% plastic strain
  auto ep2 = neml2::Scalar::full(0.1, _model->options());   // 10% plastic strain

  neml2::ValueMap in1, in2;
  in1["forces/s"] = s;
  in1["forces/ep"] = ep1;
  in2["forces/s"] = s;
  in2["forces/ep"] = ep2;

  auto out1 = _model->value(in1);
  auto out2 = _model->value(in2);

  auto rate1 = out1.at("state/internal/gamma_rate").item<double>();
  auto rate2 = out2.at("state/internal/gamma_rate").item<double>();

  // Higher plastic strain means higher yield stress,
  // so same applied stress gives lower flow rate
  EXPECT_GT(rate1, rate2);
}

#endif // NEML2_ENABLED
