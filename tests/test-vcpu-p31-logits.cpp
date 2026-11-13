// VLANIX P3.1 — actual llama_decode logits and greedy token-ID parity.
// Single local GGUF model; two sequential contexts, native and opt-in hook.
// Laboratory only. No prompt/model data emitted or committed.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Trace {
    std::vector<llama_token> chosen;
    std::vector<std::vector<float>> logits;
};

static Trace run(struct llama_model* model, const std::vector<llama_token>& prompt_tokens,
                 bool enable_hook, int predicted) {
    if (setenv("GGML_VCPU_LAB_ENABLE", enable_hook ? "1" : "0", 1) != 0)
        throw std::runtime_error("setenv failed");
    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=256;
    cp.n_batch=128;
    cp.n_ubatch=128;
    cp.n_threads=2;
    cp.n_threads_batch=2;
    struct llama_context*ctx=llama_init_from_model(model,cp);
    if(!ctx)throw std::runtime_error("context init failed");
    const llama_vocab*vocab=llama_model_get_vocab(model);
    const int num_vocab=llama_vocab_n_tokens(vocab);
    Trace result;
    std::vector<llama_token> copy=prompt_tokens;
    if(llama_decode(ctx,llama_batch_get_one(copy.data(),(int)copy.size()))!=0)
        throw std::runtime_error("prompt decode failed");
    for(int step=0;step<predicted;step++) {
        float *data=llama_get_logits_ith(ctx,-1);
        if(!data)throw std::runtime_error("no logits");
        result.logits.emplace_back(data,data+num_vocab);
        const float *begin=result.logits.back().data();
        int best=(int)(std::max_element(begin,begin+num_vocab)-begin);
        result.chosen.push_back((llama_token)best);
        llama_token token=(llama_token)best;
        if(step+1<predicted) {
            if(llama_decode(ctx,llama_batch_get_one(&token,1))!=0)
                throw std::runtime_error("greedy decode failed");
        }
    }
    llama_free(ctx);
    return result;
}

int main(int argc,char **argv) {
    try {
        if(argc!=2)throw std::runtime_error("usage: vcpu-p31-logits-parity local.gguf");
        llama_model_params mp=llama_model_default_params();
        mp.n_gpu_layers=0;
        struct llama_model*model=llama_model_load_from_file(argv[1],mp);
        if(!model)throw std::runtime_error("failed to load GGUF");
        const llama_vocab*vocab=llama_model_get_vocab(model);
        std::string prompt="A sequence of even integers: 2, 4, 6, 8,";
        std::vector<llama_token> prompt_ids(128);
        int n=llama_tokenize(vocab,prompt.c_str(),(int)prompt.size(),
                             prompt_ids.data(),(int)prompt_ids.size(),true,false);
        if(n<=0||n>128)throw std::runtime_error("prompt tokenizer failed");
        prompt_ids.resize(n);
        const int predict=16;
        auto native=run(model,prompt_ids,false,predict);
        auto custom=run(model,prompt_ids,true,predict);
        llama_model_free(model);
        if(native.chosen.size()!=custom.chosen.size())throw std::runtime_error("token count mismatch");
        double maximum=0,mean_abs_sum=0;
        size_t total=0,invalid=0;
        for(size_t step=0;step<native.logits.size();step++){
            const auto &a=native.logits[step],&b=custom.logits[step];
            if(a.size()!=b.size())throw std::runtime_error("vocab mismatch");
            for(size_t i=0;i<a.size();i++){
                // Ignore -INF values that agree; count other nonfinite values as error.
                if(a[i]==b[i])continue;
                if(!std::isfinite(a[i])||!std::isfinite(b[i])){invalid++;continue;}
                double diff=std::abs(double(a[i])-double(b[i]));
                maximum=std::max(maximum,diff);
                mean_abs_sum+=diff;
            }
            total+=a.size();
        }
        const bool token_ids_match=native.chosen==custom.chosen;
        const bool passed=token_ids_match && invalid==0 && maximum<=0.02;
        std::cout<<std::fixed<<std::setprecision(9)
                 <<"P31_LOGITS_PARITY token_ids_match="<<(token_ids_match?"true":"false")
                 <<" token_steps="<<predict
                 <<" vocab="<<(native.logits.empty()?0:native.logits[0].size())
                 <<" max_abs_diff="<<maximum
                 <<" mean_abs_diff="<<(total?mean_abs_sum/total:0)
                 <<" invalid_comparisons="<<invalid
                 <<" status="<<(passed?"PASS":"REVIEW")<<"\n";
        return passed?0:5;
    }catch(const std::exception&e){
        std::cerr<<"P31_LOGITS_ERROR "<<e.what()<<"\n";
        return 3;
    }
}
