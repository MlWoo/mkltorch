
#include <stdlib.h>
#include <string.h>

#include "THAtomic.h"
#include "luaT.h"
#include "THFile.h"
#include "THStorage.h"
#include "THTensor.h"
#include <assert.h>
#include "MKLDNN.h"

#define MKLDNN_(NAME)       TH_CONCAT_3(NAME, _, BIT)

#define torch_Tensor	    TH_CONCAT_STRING_3(torch., Real, Tensor)
#define torch_tensor	    TH_CONCAT_STRING_3(torch., Real, Tensor)
#define torch_Storage       TH_CONCAT_STRING_3(torch., Real, Storage)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_, Real, Tensor_, NAME)

#define WORKSPACE_(NAME)    TH_CONCAT_4(dnn, Real, Workspace, NAME)					
#define tensorDNNWorkspace  TH_CONCAT_3(Real, Tensor, DNNWorkspace)
#define dnnWorkspace        TH_CONCAT_3(DNN, Real, Workspace)

#define TH_MKL_(NAME)       TH_CONCAT_4(THMKL, Real, Tensor, NAME)					
#define torch_mkl_tensor    TH_CONCAT_STRING_4(torch., MKL, Real, Tensor)
#define THMKLTensor	        TH_CONCAT_3(THMKL, Real, Tensor)
#define torch_mkl_(NAME)    TH_CONCAT_4(torch_MKL, Real, Tensor_, NAME)		


#include "generic/dnnWorkspace.h"
#include "generateFloatTypes.h"

#include "generic/dnnWorkspace.c"
#include "generateFloatTypes.h"


#include "generic/tensor.h"
#include "generateAllTypes.h"

#include "generic/tensor.c"
#include "generateAllTypes.h"


