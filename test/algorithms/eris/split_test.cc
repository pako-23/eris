#include "algorithms/eris/aggregator.pb.h"
#include "algorithms/eris/split.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

class SplitTest : public testing::Test {
protected:
  SplitTest(void) : rng(time(NULL)), real{0.0, 1.0}, integer{1, 100} {}

  std::vector<float> test_parameters(size_t size) {
    std::vector<float> parameters;

    parameters.reserve(size);
    for (size_t i = 0; i < size; ++i)
      parameters.push_back(real(rng));

    return parameters;
  }

  std::default_random_engine rng;
  std::uniform_real_distribution<float> real;
  std::uniform_int_distribution<uint32_t> integer;
};

TEST_F(SplitTest, GetFragmentSizeDivisible) {
  const uint32_t splits = integer(rng);
  const uint32_t fragment_size = integer(rng);
  std::vector<float> parameters = test_parameters(splits * fragment_size);
  RandomSplit splitter;

  splitter.configure(parameters.size(), splits, 42);

  for (uint32_t i = 0; i < splits; ++i)
    ASSERT_EQ(splitter.get_fragment_size(i), fragment_size);
}

TEST_F(SplitTest, GetFragmentSizeNonDivisible) {
  const uint32_t splits = 10;
  const uint32_t fragment_size = integer(rng) + 3;
  std::vector<float> parameters = test_parameters(splits * fragment_size + 3);
  RandomSplit splitter;

  splitter.configure(parameters.size(), splits, 42);

  for (uint32_t i = 0; i < 3; ++i)
    ASSERT_EQ(splitter.get_fragment_size(i), fragment_size + 1);
  for (uint32_t i = 3; i < splits; ++i)
    ASSERT_EQ(splitter.get_fragment_size(i), fragment_size);
}

TEST_F(SplitTest, GetFragmentSizeNonDivisibleByOne) {
  const uint32_t splits = 10;
  const uint32_t fragment_size = integer(rng) + 1;
  std::vector<float> parameters = test_parameters(splits * fragment_size + 1);
  RandomSplit splitter;

  splitter.configure(parameters.size(), splits, 42);

  ASSERT_EQ(splitter.get_fragment_size(0), fragment_size + 1);
  for (uint32_t i = 1; i < splits; ++i)
    ASSERT_EQ(splitter.get_fragment_size(i), fragment_size);
}

TEST_F(SplitTest, Split) {
  const uint32_t splits = integer(rng) + 2;
  const uint32_t fragment_size = integer(rng);
  const uint32_t round = integer(rng);
  const uint32_t samples = integer(rng);
  std::vector<float> parameters = test_parameters(splits * fragment_size);

  RandomSplit splitter;

  splitter.configure(parameters.size(), splits, 42);
  std::vector<eris::WeightSubmissionRequest> fragments =
      splitter.split(std::make_pair(parameters, samples), round);

  size_t total_size = 0;
  for (const auto &fragment : fragments) {
    ASSERT_EQ(fragment.round(), round);
    ASSERT_EQ(fragment.samples(), samples);
    ASSERT_EQ(fragment.weight_size(), fragment_size);
    total_size += fragment.weight_size();
  }
  ASSERT_EQ(total_size, parameters.size());

  bool different = false;
  size_t i = 0;
  for (const auto &fragment : fragments)
    for (int j = 0; j < fragment.weight_size(); ++j)
      if (parameters[i] != fragment.weight(j))
        different = true;

  EXPECT_TRUE(different);
}

TEST_F(SplitTest, SplitSameSeed) {
  const uint32_t splits = integer(rng) + 2;
  const uint32_t fragment_size = integer(rng);
  const uint32_t round = integer(rng);
  const uint32_t samples = integer(rng);
  std::vector<float> parameters = test_parameters(splits * fragment_size);

  RandomSplit first_splitter;

  first_splitter.configure(parameters.size(), splits, 42);
  std::vector<eris::WeightSubmissionRequest> first_fragments =
      first_splitter.split(std::make_pair(parameters, samples), round);

  RandomSplit second_splitter;

  second_splitter.configure(parameters.size(), splits, 42);
  std::vector<eris::WeightSubmissionRequest> second_fragments =
      second_splitter.split(std::make_pair(parameters, samples), round);

  EXPECT_EQ(first_fragments.size(), second_fragments.size());
  for (size_t i = 0; i < first_fragments.size(); ++i)
    for (int j = 0; j < first_fragments[i].weight_size(); ++j)
      EXPECT_EQ(first_fragments[i].weight(j), second_fragments[i].weight(j));
}

TEST_F(SplitTest, SplitDifferentSeed) {
  const uint32_t splits = integer(rng) + 2;
  const uint32_t fragment_size = integer(rng);
  const uint32_t round = integer(rng);
  const uint32_t samples = integer(rng);
  std::vector<float> parameters = test_parameters(splits * fragment_size);

  RandomSplit first_splitter;

  first_splitter.configure(parameters.size(), splits, 42);
  std::vector<eris::WeightSubmissionRequest> first_fragments =
      first_splitter.split(std::make_pair(parameters, samples), round);

  RandomSplit second_splitter;

  second_splitter.configure(parameters.size(), splits, 100);
  std::vector<eris::WeightSubmissionRequest> second_fragments =
      second_splitter.split(std::make_pair(parameters, samples), round);

  EXPECT_EQ(first_fragments.size(), second_fragments.size());
  bool difference = false;
  for (size_t i = 0; i < first_fragments.size(); ++i)
    for (int j = 0; j < first_fragments[i].weight_size(); ++j)
      if (first_fragments[i].weight(j) != second_fragments[i].weight(j))
        difference = true;

  EXPECT_TRUE(difference);
}

TEST_F(SplitTest, Reassemble) {
  const uint32_t splits = integer(rng) + 2;
  const uint32_t fragment_size = 1; // integer(rng);
  const uint32_t round = integer(rng);
  const uint32_t samples = integer(rng);
  std::vector<float> parameters = test_parameters(splits * fragment_size);

  RandomSplit splitter;

  splitter.configure(parameters.size(), splits, 42);
  std::vector<eris::WeightSubmissionRequest> fragments =
      splitter.split(std::make_pair(parameters, samples), round);

  std::vector<eris::WeightUpdate> updates(fragments.size());
  for (size_t i = 0; i < updates.size(); ++i) {
    updates[i].set_round(round);

    for (int j = 0; j < fragments[i].weight_size(); ++j)
      updates[i].add_weight(fragments[i].weight(j));
  }

  std::vector<float> reassembled = splitter.reassemble(updates);

  EXPECT_EQ(parameters.size(), reassembled.size());
  for (size_t i = 0; i < parameters.size(); ++i)
    EXPECT_NEAR(parameters[i], reassembled[i],
                5 * std::numeric_limits<float>::epsilon());
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  google::protobuf::ShutdownProtobufLibrary();
  return ret;
}
