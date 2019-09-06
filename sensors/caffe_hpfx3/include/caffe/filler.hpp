// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->num();
    CHECK(dim);
    for (int i = 0; i < blob->num(); ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        data[i * dim + j] /= sum;
      }
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
 *        set inversely proportional to number of incoming nodes, outgoing
 *        nodes, or their average.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks.
 *
 * It fills the incoming matrix by randomly sampling uniform data from [-scale,
 * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
 * average, depending on the variance_norm option. You should make sure the
 * input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
 * = fan_out. Note that this is currently not the case for inner product layers.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype scale = sqrt(Dtype(3) / n);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

//add by yaoye
template <typename Dtype>
class QianFiller : public Filler<Dtype> {
 public:
  explicit QianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
	Dtype* weights = blob->mutable_cpu_data();
	weights[0] = -1.0/12.0;
	weights[1] = 2.0/12.0;
	weights[2] = -2.0/12.0;
	weights[3] = 2.0/12.0;
	weights[4] = -1.0/12.0;
	weights[5] = 2.0/12.0;
	weights[6] = -6.0/12.0;
	weights[7] = 8.0/12.0;
	weights[8] = -6.0/12.0;
	weights[9] = 2.0/12.0;
	weights[10] = -2.0/12.0;
	weights[11] = 8.0/12.0;
	weights[12] = -12.0/12.0;
	weights[13] = 8.0/12.0;
	weights[14] = -2.0/12.0;
	weights[15] = 2.0/12.0;
	weights[16] = -6.0/12.0;
	weights[17] = 8.0/12.0;
	weights[18] = -6.0/12.0;
	weights[19] = 2.0/12.0;
	weights[20] = -1.0/12.0;
	weights[21] = 2.0/12.0;
	weights[22] = -2.0/12.0;
	weights[23] = 2.0/12.0;
	weights[24] = -1.0/12.0;

  //5x5 filter2
  weights[25] = 0;
  weights[26] = 0;
  weights[27] = 0;
  weights[28] = 0;
  weights[29] = 0;
  weights[30] = 0;
  weights[31] = 0;
  weights[32] = -1.0/4.0;
  weights[33] = 0;
  weights[34] = 0;
  weights[35] = 0;
  weights[36] = -1.0/4.0;
  weights[37] =  1.0;
  weights[38] = -1.0/4.0;
  weights[39] = 0;
  weights[40] = 0;
  weights[41] = 0;
  weights[42] = -1.0/4.0;
  weights[43] = 0;
  weights[44] = 0;
  weights[45] = 0;
  weights[46] = 0;
  weights[47] = 0;
  weights[48] = 0;
  weights[49] = 0;

  //5x5 filter3
  weights[50] = 0;
  weights[51] = 0;
  weights[52] = 0;
  weights[53] = 0;
  weights[54] = 0;
  weights[55] = 0;
  weights[56] = -1.0/4.0;
  weights[57] =  2.0/4.0;
  weights[58] = -1.0/4.0;
  weights[59] = 0;
  weights[60] = 0;
  weights[61] =  2.0/4.0;
  weights[62] = -1.0;
  weights[63] =  2.0/4.0;
  weights[64] = 0;
  weights[65] = 0;
  weights[66] = -1.0/4.0;
  weights[67] =  2.0/4.0;
  weights[68] = -1.0/4.0;
  weights[69] = 0;
  weights[70] = 0;
  weights[71] = 0;
  weights[72] = 0;
  weights[73] = 0;
  weights[74] = 0;    

  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where
 *        @f$ \sigma^2 @f$ is set inversely proportional to number of incoming
 *        nodes, outgoing nodes, or their average.
 *
 * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
 * accounts for ReLU nonlinearities.
 *
 * Aside: for another perspective on the scaling factor, see the derivation of
 * [Saxe, McClelland, and Ganguli 2013 (v3)].
 *
 * It fills the incoming matrix by randomly sampling Gaussian data with std =
 * sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
 * the variance_norm option. You should make sure the input blob has shape (num,
 * a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
 * is currently not the case for inner product layers.
 */
template <typename Dtype>
class MSRAFiller : public Filler<Dtype> {
 public:
  explicit MSRAFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype std = sqrt(Dtype(2) / n);
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/*!
@brief Fills a Blob with coefficients for bilinear interpolation.

A common use case is with the DeconvolutionLayer acting as upsampling.
You can upsample a feature map with shape of (B, C, H, W) by any integer factor
using the following proto.
\code
layer {
  name: "upsample", type: "Deconvolution"
  bottom: "{{bottom_name}}" top: "{{top_name}}"
  convolution_param {
    kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
    num_output: {{C}} group: {{C}}
    pad: {{ceil((factor - 1) / 2.)}}
    weight_filler: { type: "bilinear" } bias_term: false
  }
  param { lr_mult: 0 decay_mult: 0 }
}
\endcode
Please use this by replacing `{{}}` with your values. By specifying
`num_output: {{C}} group: {{C}}`, it behaves as
channel-wise convolution. The filter shape of this deconvolution layer will be
(C, 1, K, K) where K is `kernel_size`, and this filler will set a (K, K)
interpolation kernel for every channel of the filter identically. The resulting
shape of the top feature map will be (B, C, factor * H, factor * W).
Note that the learning rate and the
weight decay are set to 0 in order to keep coefficient values of bilinear
interpolation unchanged during training. If you apply this to an image, this
operation is equivalent to the following call in Python with Scikit.Image.
\code{.py}
out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
\endcode
 */
template <typename Dtype>
class BilinearFiller : public Filler<Dtype> {
 public:
  explicit BilinearFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();
    int f = ceil(blob->width() / 2.);
    Dtype c = (blob->width() - 1) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) {
      Dtype x = i % blob->width();
      Dtype y = (i / blob->width()) % blob->height();
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
  } else if (type == "qian") {
    return new QianFiller<Dtype>(param);
  } else if (type == "msra") {
    return new MSRAFiller<Dtype>(param);
  } else if (type == "bilinear") {
    return new BilinearFiller<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
