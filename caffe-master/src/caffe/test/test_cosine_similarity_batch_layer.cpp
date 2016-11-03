#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cosine_similarity_batch_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CosineSimilarityBatchLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CosineSimilarityBatchLayerTest()
      : blob_bottom_data_features(new Blob<Dtype>(5, 10, 1, 1)),
        blob_bottom_data_labels(new Blob<Dtype>(5, 1, 1, 1)),
        blob_top_sim(new Blob<Dtype>(45, 1, 1, 1)),
        blob_top_labels(new Blob<Dtype>(45, 1, 1, 1))

  {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_mean(1.0);
    filler_param.set_std(0.01); 
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_features);
    blob_bottom_vec_.push_back(blob_bottom_data_features);
    
    for (int i = 0; i < blob_bottom_data_labels->count(); ++i) {
      int r = caffe_rng_rand() % 10;
      blob_bottom_data_labels->mutable_cpu_data()[i] = r;  // -1 or 1
    } 
    blob_bottom_vec_.push_back(blob_bottom_data_labels);
    
    blob_top_vec_.push_back(blob_top_sim);
    blob_top_vec_.push_back(blob_top_labels);
  }

  virtual ~CosineSimilarityBatchLayerTest() {
    delete blob_bottom_data_features;
    delete blob_bottom_data_labels;
    delete blob_top_sim;
    delete blob_top_labels;
  }

  Blob<Dtype>* const blob_bottom_data_features;
  Blob<Dtype>* const blob_bottom_data_labels;
  Blob<Dtype>* const blob_top_sim;
  Blob<Dtype>* const blob_top_labels;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CosineSimilarityBatchLayerTest, TestDtypesAndDevices);

TYPED_TEST(CosineSimilarityBatchLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CosineSimilarityBatchLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
 
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
     (this->blob_top_vec_), 0);
}

}  // namespace caffe
