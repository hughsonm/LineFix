#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>

static double random_double(double min,double max){
    double x{
        ((double)std::rand())/((double)RAND_MAX)
    };
    x *= (max-min);
    x += min;
    return(x);
}

class Synapse;
class Neuron;
class Layer;
class Net;

class Synapse{
public:
    double dFdw{0};
    Neuron * source{nullptr};
    Neuron * destination{nullptr};
    double weight{random_double(-1,1)};
    void show(int depth);
};
void Synapse::show(int depth){
    // std::cerr << "src:    " << source << std::endl;
    // std::cerr << "dest:   " << destination << std::endl;
    for(int ii{0}; ii<depth; ++ii) std::cerr << "\t";
    std::cerr << "weight: " << weight << std::endl;
}

class Neuron{
public:
    double dFdg{0};
    double dFdb{0};
    std::vector<Synapse *> inputs;
    double z{0.0};
    double g{0.5};
    double gminus{0.5};
    double gprime{0.25};
    double b{random_double(-1,1)};
    void calculate_output(void){
        z = b;
        for(const auto& syn: inputs){
            if(syn != nullptr){
                Neuron * src_neuron = syn->source;
                if(src_neuron != nullptr){
                    z += src_neuron->g*syn->weight;
                }
            }
        }
        g = 1.0/(1+std::exp(-(z)));
        gminus = 1.0/(1+std::exp((z)));
        gprime = g*gminus;
    }
    Synapse * connect(Neuron * new_neuron);
    ~Neuron(){
        for(auto& syn : inputs){
            delete syn;
        }
    }
    void show(int depth);
};

void Neuron::show(int depth){
    std::string indent{""};
    for(int ii{0}; ii<depth; ++ii) indent += "\t";
    std::cerr << indent << "This neuron has " << inputs.size() << " inputs." << std::endl;
    std::cerr << indent << "My offset is " << b << std::endl;
    std::cerr << indent << "My current g is " << g << std::endl;
    for(auto ii{0}; ii<inputs.size();++ii){
        inputs[ii]->show(depth+1);
    }
}

Synapse * Neuron::connect(Neuron * new_neuron){
    Synapse * syn{new Synapse};
    syn->source = new_neuron;
    syn->destination = this;
    inputs.push_back(syn);
    return(syn);
}

class Layer{
public:
    Layer * previous{nullptr};
    Layer * next{nullptr};
    std::vector<Neuron *> neurons;
    void stimulate(void){
        for(auto& neuron : neurons){
            if(neuron) neuron->calculate_output();
        }
    }
    void add_neuron(void);
    void assign_neural_values(
        const std::vector<double> & assignment_values
    );
    ~Layer(){
        for(auto& neuron : neurons){
            delete neuron;
        }
    }
    void propagate_derivative(void);
    void clear_derivative(void);
    void show(int depth);
};

void Layer::clear_derivative(void){
    for(auto& neuron : neurons){
        neuron->dFdg = 0;
    }
}

void Layer::propagate_derivative(void){
    if(previous){
        previous->clear_derivative();
        for(auto& neuron : neurons){
            // Add (dFdme * dmedsum * dsumdprev) to my prev's derivative.
            for(auto& syn : neuron->inputs){
                syn->source->dFdg += neuron->dFdg*neuron->gprime*syn->weight;
                syn->dFdw = neuron->dFdg*neuron->gprime*syn->source->g;
            }
            // Calculate dF w.r.t. my offset
            neuron->dFdb = neuron->dFdg*neuron->gprime;
        }
        previous->propagate_derivative();
    }
}


void Layer::show(int depth){
    std::string indent{""};
    for(int ii{0}; ii<depth; ++ii) indent += "\t";
    std::cerr << indent << "This layer has " << neurons.size() << " neurons." << std::endl;

    for(auto ii{0}; ii<neurons.size(); ++ii)
    {
        std::cerr << indent << "Neuron Number: " << ii << std::endl;
        neurons[ii]->show(depth+1);
    }
}

void Layer::assign_neural_values(
    const std::vector<double> & assignment_values
){
    assert(assignment_values.size()==neurons.size());
    for(auto ii{0}; ii<neurons.size(); ++ii){
        neurons[ii]->g = assignment_values[ii];
    }
}

void Layer::add_neuron(void){
    Neuron * nn{new Neuron};
    // Connect this neuron to all in previous layer.
    if(previous){
        for(auto& prev_neuron : previous->neurons){
            nn->connect(prev_neuron);
        }
    }
    // Connect this neuron to all in next layer.
    if(next){
        for(auto& next_neuron : next->neurons){
            next_neuron->connect(nn);
        }
    }
    neurons.push_back(nn);
}

class Net{
public:
    std::vector<Layer *> layers;

    void predict(
        const std::vector<double> & input_vector,
        std::vector<double> & output_vector
    );
    void back_prop(
        const std::vector<double> & input_example,
        const std::vector<double> & output_example
    );
    void update_weights(double step_size);
    Layer * add_layer(
        int n_neurons
    );
    void show(void);
    int cull(double threshold);
    ~Net(void){
        for(auto& layer:layers){
            delete layer;
        }
    }
};

int Net::cull(
    double threshold
){
    int kill_count{0};
    for(auto& layer : layers){
        for(auto& neuron : layer->neurons){
            auto syn_pick{0},syn_place{0};
            for(auto& syn : neuron->inputs){
                double abs_weight{std::abs(syn->weight)};
                if(abs_weight < threshold){
                    delete syn;
                    kill_count++;
                }else{
                    neuron->inputs[syn_place++] = neuron->inputs[syn_pick];
                }
                syn_pick++;
            }
            neuron->inputs.resize(syn_place);
        }
    }
    return(kill_count);
}

void Net::update_weights(double step_size){
    for(auto& layer : layers){
        for(auto& neuron : layer->neurons){
            for(auto& syn : neuron->inputs){
                syn->weight -= syn->dFdw*step_size;
            }
            neuron->b -= neuron->dFdb*step_size;
        }
    }
}

void Net::back_prop(
    const std::vector<double> & input_example,
    const std::vector<double> & output_example
){
    assert(layers.size());
    std::vector<Neuron *> & final_neurons(layers.back()->neurons);
    assert(final_neurons.size() == output_example.size());
    std::vector<double> diff(output_example.size());
    for(auto ii{0}; ii < output_example.size();++ii){
        diff[ii] = output_example[ii]-final_neurons[ii]->g;
    }
    //  For each example
    //  Hand-calc the derivatives for
    for(auto ii{0}; ii<final_neurons.size(); ++ii){
        final_neurons[ii]->dFdg = -2*diff[ii];
    }
    layers.back()->propagate_derivative();
}



void Net::show(void){
    std::cerr << "This neural net has " << layers.size() << " layers." << std::endl;
    int neuron_depth = 1;
    if(layers.size()){
        std::cerr << "Let us show all the layers" << std::endl;
        Layer * current_layer{layers[0]};
        bool keep_printing{true};
        bool found_a_neuron{false};
        while(keep_printing){
            // std::cerr << "\tThis layer has " << current_layer->neurons.size() << " neurons" << std::endl;
            if(neuron_depth <= current_layer->neurons.size()){
                std::cerr << "*\t";
                found_a_neuron = true;
            } else{
                std::cerr <<"\t";
            }
            if(current_layer->next){
                // std::cerr << "Moving to next layer" << std::endl;
                current_layer = current_layer->next;
            } else{
                // std::cerr << "Moving back to the start..." << std::endl;
                current_layer = layers[0];
                std::cerr << std::endl;
                keep_printing = found_a_neuron;
                found_a_neuron = false;
                neuron_depth++;
            }
        }
    }
    for(auto ii{0};ii<layers.size();++ii)
    {
        std::cerr << "Layer Number: " << ii << std::endl;
        layers[ii]->show(1);
    }
}

void Net::predict(
    const std::vector<double> & input_vector,
    std::vector<double> & output_vector
){
    if(layers.size()){
        // Assign values to the first neurons
        layers[0]->assign_neural_values(input_vector);

        // Stimulate next neurons
        for(auto ii{1}; ii<layers.size();++ii)
        {
            layers[ii]->stimulate();
        }

        // Read off values from final neuron.
        const std::vector<Neuron*>& final_neurons{layers.back()->neurons};
        output_vector.resize(final_neurons.size());
        for(auto ii{0}; ii<final_neurons.size();++ii){
            output_vector[ii] = final_neurons[ii]->g;
        }
    }
}

Layer * Net::add_layer(
    int n_neurons
){
    Layer * ll{new Layer};

    if(layers.size()){
        ll->previous = layers.back();
        layers.back()->next = ll;
    }

    for(auto ii{0}; ii<n_neurons; ++ii){
        ll->add_neuron();
    }
    layers.push_back(ll);
    return(ll);
}

int main(int argc, char**argv)
{


    std::ifstream reader("CleanedShort.txt");
    std::string word;
    std::map<std::string,int> word2index;
    std::map<std::string,double> word2freq;
    std::vector<std::string> index2word;
    std::vector<int> story;
    int current_index{0};
    int word_count{0};
    while(!reader.eof()){
        reader >> word;
        word_count++;
        if(!word2index.count(word)){
            word2index[word] = current_index++;
            word2freq[word] = 1.0;
            index2word.push_back(word);
        }
        word2freq[word] += 1.0;
        story.push_back(word2index[word]);
        // std::cerr << word << std::endl;
    }
    int word_index{0};
    for(auto& word : index2word){
        word2freq[word] /= word_count;
        std::cerr << word_index++ << "\t" << word << "\t" << word2freq[word] << std::endl;
    }
    std::cerr << "Saw " << word_count << " words, and made " << current_index << " map entries " << std::endl;
    reader.close();

    int n_entries{current_index};

    Net brain;
    brain.add_layer(n_entries);
    brain.add_layer(n_entries/10);
    brain.add_layer(n_entries);

    std::vector<double> input_vector(n_entries);
    std::vector<double> desired_output_vector(n_entries);
    std::vector<double> result_vector(n_entries);


    bool keep_interacting{true};

    while(keep_interacting){
        std::cout << "Train(t), Cull(c), Map(m), Vector(v), Show(s), Quit(q)" << std::endl;
        std::string request;
        std::cin >> request;
        if(request=="t"){
            std::cerr << "How big shoudl the step size be?" << std::endl;
            double step_size;
            std::cin >> step_size;
            std::cerr << "How many samples shall we run?" << std::endl;
            int n_samples;
            std::cin >> n_samples;
            std::cerr << "How many words per sample?" << std::endl;
            int sample_size;
            std::cin >> sample_size;
            std::cerr << "Trianing with step size = " << step_size << std::endl;
            std::cerr << "**********" << std::endl;
            std::fill(input_vector.begin(),input_vector.end(),0.0);
            std::fill(desired_output_vector.begin(),desired_output_vector.end(),0.0);
            std::vector<int> sample(2*sample_size+1);
            for(auto isample{0}; isample<n_samples; ++isample){
                auto choose_mid{
                    sample_size + (std::rand() % (story.size()-2*sample_size))
                };
                std::cerr << "Sample number " << isample << ": ";
                for(auto ii{0}; ii<(2*sample_size+1);++ii){
                    sample[ii] = story[choose_mid-sample_size+ii];
                    std::cerr << index2word[sample[ii]] << " ";
                }
                std::cerr << std::endl;
                for(auto & right_idx : sample){
                    desired_output_vector[right_idx] = 1;
                }
                for(auto& left_idx : sample){
                    input_vector[left_idx] = 1;
                    double step_for_pair = step_size/word2freq[index2word[left_idx]]/word_count;
                    brain.predict(
                        input_vector,
                        result_vector
                    );
                    brain.back_prop(
                        input_vector,
                        desired_output_vector
                    );
                    brain.update_weights(step_for_pair);
                    input_vector[left_idx] = 0;
                }
                for(auto & right_idx : sample){
                    desired_output_vector[right_idx] = 0;
                }
            }
        }else if(request == "c"){
            double thresh;
            std::cerr << "Cull to what threshold?" << std::endl;
            std::cin >> thresh;
            std::cerr << "Culling with threshold = " << thresh << std::endl;
            std::cerr << "Removed " << brain.cull(thresh) << " synapses." << std::endl;

        }else if(request == "m"){
            std::cerr << "Which word would you like to investigate?" << std::endl;
            std::string in_word;
            std::cin >> in_word;
            if(word2index.count(in_word)){
                int in_word_index{word2index[in_word]};
                input_vector[in_word_index] = 1;
                brain.predict(
                    input_vector,
                    result_vector
                );
                std::vector<int> idcs_to_sort(result_vector.size());
                for(auto ii{0};ii<idcs_to_sort.size();++ii){
                    idcs_to_sort[ii] = ii;
                }
                std::cerr << "Word count = " << word_count;
                std::sort(
                    idcs_to_sort.begin(),idcs_to_sort.end(),[&result_vector](int x,int y){
                        return(result_vector[x]>result_vector[y]);
                    }
                );
                for(auto ii{0};ii<10;++ii){
                    std::cerr << "Request <" << in_word << "> maps to result <" << index2word[idcs_to_sort[ii]] << "> with " << 100.0*result_vector[idcs_to_sort[ii]] << " confidence." << std::endl;
                }               
            }else{
                std::cerr << "Request: <" << request << "> does not appear in dictionary" << std::endl;
            }
        }else if(request == "v"){
            std::cerr << "Which word would you like to investigate?" << std::endl;
            std::string v_word;
            std::cin >> v_word;
            if(word2index.count(v_word)){
                int v_word_index{word2index[v_word]};
                input_vector[v_word_index] = 1;
                brain.predict(
                    input_vector,
                    result_vector
                );
                std::cerr << "<";
                for(auto& neuron : brain.layers[1]->neurons){
                    std::cerr << neuron->g << ",";
                }
                std::cerr << ">" << std::endl;;
            }else{
                std::cerr << "Request: <" << v_word << "> does not appear in dictionary" << std::endl;
            }
        }else if(request == "s"){
            brain.show();
        }else if(request == "q"){
            keep_interacting = false;
        }
    }


    return 0;
}
