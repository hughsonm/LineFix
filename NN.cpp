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
    // std::cout << "src:    " << source << "\n";
    // std::cout << "dest:   " << destination << "\n";
    for(int ii{0}; ii<depth; ++ii) std::cout << "\t";
    std::cout << "weight: " << weight << "\n";
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
    std::cout << indent << "This neuron has " << inputs.size() << " inputs." << "\n";
    std::cout << indent << "My offset is " << b << "\n";
    std::cout << indent << "My current g is " << g << "\n";
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
    std::cout << indent << "This layer has " << neurons.size() << " neurons." << "\n";

    for(auto ii{0}; ii<neurons.size(); ++ii)
    {
        std::cout << indent << "Neuron Number: " << ii << "\n";
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
    double get_grad_norm(void){
        double norm_sq{0};
        for(auto& layer : layers){
            for(auto& neuron:layer->neurons){
                for(auto& syn:neuron->inputs){
                    norm_sq += (syn->dFdw)*(syn->dFdw);
                }
                norm_sq += (neuron->dFdb)*(neuron->dFdb);
            }
        }
        return(std::sqrt(norm_sq));
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
    std::cout << "This neural net has " << layers.size() << " layers." << "\n";
    int neuron_depth = 1;
    if(layers.size()){
        std::cout << "Let us show all the layers" << "\n";
        Layer * current_layer{layers[0]};
        bool keep_printing{true};
        bool found_a_neuron{false};
        while(keep_printing){
            // std::cout << "\tThis layer has " << current_layer->neurons.size() << " neurons" << "\n";
            if(neuron_depth <= current_layer->neurons.size()){
                std::cout << "*\t";
                found_a_neuron = true;
            } else{
                std::cout <<"\t";
            }
            if(current_layer->next){
                // std::cout << "Moving to next layer" << "\n";
                current_layer = current_layer->next;
            } else{
                // std::cout << "Moving back to the start..." << "\n";
                current_layer = layers[0];
                std::cout << "\n";
                keep_printing = found_a_neuron;
                found_a_neuron = false;
                neuron_depth++;
            }
        }
    }
    for(auto ii{0};ii<layers.size();++ii)
    {
        std::cout << "Layer Number: " << ii << "\n";
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

class Game{
public:
    std::string home_team{""},away_team{""};
    int home_score{0},away_score{0};
    int home_rest{0},away_rest{0};
    double exp_home_win_margin{0};
    double act_home_win_margin{0};
    bool home_covered_spread;
    void fill_input_vector(
        std::vector<double>&in_vec,
        int n_teams,
        const std::map<std::string,int>& t2i
    ){
        std::fill(
            in_vec.begin(),
            in_vec.end(),
            0.0
        );
        in_vec[t2i.at(home_team)]=1;
        in_vec[n_teams+t2i.at(away_team)]=1;
        in_vec[2*n_teams+0] = home_rest;
        in_vec[2*n_teams+1] = away_rest;
        in_vec[2*n_teams+2] = exp_home_win_margin;
    }
};

int main(int argc, char**argv)
{
    std::ifstream reader("Games.txt");
    std::string word;
    std::map<std::string,int> team2index;
    std::vector<std::string> index2team;
    std::vector<Game> all_games;
    std::string site;
    int team_name_index{0};
    while(!reader.eof()){
        // Get info for this game
        Game game_from_line;
        reader >> site;
        if(reader.eof()) break;
        if(site=="home"){
            // Each game appears in the database twice: once as home and once as away.
            // Only keep track of the home version
            reader >> game_from_line.home_team;
            reader >> game_from_line.home_score;
            reader >> game_from_line.home_rest;
            reader >> game_from_line.away_team;
            reader >> game_from_line.away_score;
            reader >> game_from_line.away_rest;
            double line;
            reader >> line;
            game_from_line.exp_home_win_margin = -line;
        } else {
            continue;
        }
        game_from_line.act_home_win_margin = game_from_line.home_score-game_from_line.away_score;
        game_from_line.home_covered_spread = (game_from_line.exp_home_win_margin<game_from_line.act_home_win_margin);
        if(!team2index.count(game_from_line.home_team)){
            std::cout << "Index:\t" << team_name_index << "\tTeam:\t";
            std::cout << game_from_line.home_team << "\n";
            team2index[game_from_line.home_team] = team_name_index++;
            index2team.push_back(game_from_line.home_team);
        }
        if(!team2index.count(game_from_line.away_team)){
            std::cout << "Index:\t" << team_name_index << "\tTeam:\t";
            std::cout << game_from_line.away_team << "\n";
            team2index[game_from_line.away_team] = team_name_index++;
            index2team.push_back(game_from_line.away_team);
        }
        all_games.push_back(game_from_line);
    }
    reader.close();

    std::vector<Game> games_test;
    std::vector<Game> games_train;
    for(auto& game : all_games){
        if(random_double(0,1)<0.9){
            games_train.push_back(game);
        } else{
            games_test.push_back(game);
        }
    }

    std::cout << "Training Set: " << games_train.size() << " games\n";
    std::cout << "Testing Set:  " << games_test.size() << " games\n";

    int n_teams{team_name_index};

    Net brain;

    std::vector<double> input_vector(2*n_teams+3);
    std::vector<double> result_vector(1);
    std::vector<double> desired_output_vector(result_vector.size());

    brain.add_layer(input_vector.size());
    brain.add_layer(n_teams*n_teams);
    brain.add_layer(result_vector.size());

    bool keep_interacting{true};
    bool show_output{true};

    while(keep_interacting){
        std::cout << "Train(t), Cull(c), Map(m), Show(s), Verify(v), Show/Hide Output(h), Quit(q)" << "\n";
        std::string request;
        std::cin >> request;
        if(request=="t"){
            std::cout << "How big should the step size be?" << "\n";
            double step_size;
            std::cin >> step_size;
            std::cout << "How many samples shall we run?" << "\n";
            int n_samples;
            std::cin >> n_samples;
            std::fill(
                desired_output_vector.begin(),
                desired_output_vector.end(),
                0.0
            );
            for(auto isample{0}; isample<n_samples; ++isample){

                const auto sample_index{std::rand()%games_train.size()};
                auto& sample_game{games_train[sample_index]};
                desired_output_vector[0] = sample_game.home_covered_spread?1.0:0.0;
                sample_game.fill_input_vector(
                    input_vector,
                    n_teams,
                    team2index
                );
                brain.predict(
                    input_vector,
                    result_vector
                );
                double prediction_error{0};
                for(auto nn{0}; nn<result_vector.size();++nn){
                    auto diff{result_vector[nn]-desired_output_vector[nn]};
                    prediction_error += diff*diff;
                }
                bool predict_home_cov{0.5 < result_vector[0]};
                brain.back_prop(
                    input_vector,
                    desired_output_vector
                );
                double grad_norm{brain.get_grad_norm()};
                brain.update_weights(step_size*prediction_error/grad_norm);
                if(show_output){
                    std::cout << "<";
                    for(auto& vi : input_vector){
                        std::cout << vi << ",";
                    }
                    std::cout << ">\n";
                    std::cout << sample_game.home_team << " vs " << sample_game.away_team << ": " << sample_game.home_score << "-" << sample_game.away_score << "(" << sample_game.exp_home_win_margin << ")" "\n";
                    if(predict_home_cov){
                        std::cout << "+";
                    } else
                    {
                        std::cout << "-";
                    }
                    if(sample_game.home_covered_spread){
                        std::cout << "+";
                    } else{
                        std::cout << "-";
                    }
                    std::cout << "\n";
                    std::cout << "Prediction error: " << prediction_error << "\n";
                    std::cout << "Gradient norm   : " << grad_norm << "\n";
                    std::cout << "Update size     : " << step_size*prediction_error/grad_norm << "\n";
                    std::cout << "\n";

                }
                std::fill(
                    desired_output_vector.begin(),
                    desired_output_vector.end(),
                    0.0
                );
            }
            std::cout << "\n";
        }else if(request == "c"){
            double thresh;
            std::cout << "Cull to what threshold?" << "\n";
            std::cin >> thresh;
            std::cout << "Culling with threshold = " << thresh << "\n";
            std::cout << "Removed " << brain.cull(thresh) << " synapses." << "\n";

        }else if(request == "m"){
            Game map_game;
            std::cout << "Which home team would you like to investigate? (Use integer index)" << "\n";
            int home_index;
            std::cin >> home_index;

            std::cout << "Which away team would you like to investigate? (Use integer index)" << "\n";
            int away_index;
            std::cin >> away_index;

            std::cout << "Amount of home team rest?" << "\n";
            std::cin >> map_game.home_rest;

            std::cout << "Amount of away team rest?" << "\n";
            std::cin >> map_game.away_rest;

            std::cout << "Expected win margin for home team?" << "\n";
            std::cin >> map_game.exp_home_win_margin;

            if(
                (0<=home_index) &&
                (home_index<n_teams) &&
                (0<=away_index) &&
                (away_index<n_teams) &&
                (0<=map_game.home_rest) &&
                (0<=map_game.away_rest)
            ){
                map_game.home_team = index2team[home_index];
                map_game.away_team = index2team[away_index];
                map_game.fill_input_vector(
                    input_vector,
                    n_teams,
                    team2index
                );

                brain.predict(
                    input_vector,
                    result_vector
                );
		std::cout << index2team[home_index] << " hosting " << index2team[away_index] << ":\n";
                std::cout << "Home cover chance: (" << index2team[home_index] <<")" << result_vector[0] << ":\n";
            }else{
                std::cout << "Invalid team indices\n";
            }
        }else if(request == "s"){
            brain.show();
        }else if(request == "q"){
            keep_interacting = false;
        }else if(request == "v"){
            auto total_n_correct{0};
            for(auto& game : games_test){
                game.fill_input_vector(
                    input_vector,
                    n_teams,
                    team2index
                );
                brain.predict(
                    input_vector,
                    result_vector
                );
                if(show_output){
                    std::cout << game.home_team << " vs " << game.away_team;
                    std::cout << ": " << game.home_score << "-";
                    std::cout << game.away_score << "(";
                    std::cout << game.exp_home_win_margin <<")\n";
                    std::cout << "Home cover chance: (" << game.home_team <<")";
                    std::cout << result_vector[0] << ":\n";
                }
                bool predict_home_coverage{0.5<result_vector[0]};
                if(predict_home_coverage == game.home_covered_spread){
                    total_n_correct++;
                }
            }
            std::cout << games_test.size() << " total tests" << "\n";
            std::cout << total_n_correct << " correct" << "\n";
        } else if(request == "h"){
            show_output = !show_output;
        }
    }


    return 0;
}
