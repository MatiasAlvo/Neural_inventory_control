
from shared_imports import *


class FullyConnectedForecaster(nn.Module):
    """
    Fully connected neural network
    """

    def __init__(self, neurons_per_hidden_layer, lead_times, qs=np.arange(0.05, 1, 0.05), activation_function=nn.ELU(), device=None):
        """
        Arguments:
            neurons_per_hidden_layer: list
                list of integers, where each integer is the number of neurons in a hidden layer
            lead_times: list
                list of integers, where each integer represents a lead time. 
                However, recall that for each lead time, we predict cummulative demand for the next (lead_time + 1) weeks
            qs: list
                list of quantiles for predicting the cumulative demand over the lead times
        """

        super().__init__() # Initialize super class
        self.qs = qs.round(2)  # List of quantiles
        self.qs_dict = {round(q, 2): i for i, q in enumerate(qs)} # Dictionary of quantiles and their indices
        lead_times = torch.tensor(lead_times).int()
        self.lead_times = lead_times # List of lead times
        self.min_lead_time = min(lead_times)
        self.lead_times_dict = {lead_time: i for i, lead_time in enumerate(lead_times)} # Dictionary of lead times and their indices

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Add 0 and 1 to the list of quantiles
        self.prob_points = torch.tensor([0] + list(self.qs) + [1]).to(self.device)

        # Get a unique name for the model (which for now is the timestamp at the moment of creation)
        self.name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Define layers
        self.activation_function = activation_function
        self.layers = []
        for output_neurons in neurons_per_hidden_layer:
            self.layers.append(nn.LazyLinear(output_neurons))
            self.layers.append(self.activation_function)
            
        self.layers.append(nn.LazyLinear(len(qs)*len(self.lead_times)))
        
        # Define network as a sequence of layers
        self.net = nn.Sequential(*self.layers)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, x):
        """
        Forward pass
        """
        
        x = self.net(x)

        return torch.clip(x, min=0).reshape(*x.shape[:-1], len(self.qs), len(self.lead_times)) # clip output to be non-negative
    
    def get_quantile(self, x, quantile, lead_times):
        """
        Get quantile from predicted distribution
        Parameters:
        -----------
        x: tensor
            feature tensor of shape (batch_size, num_stores, num_features)
        quantile: tensor
            tensor of quantiles between 0 and 1 with shape (batch_size, num_stores) 
        lead_time_per_sample: tensor
            tensor of lead times with shape (batch_size, num_stores)
        """

        # For each entry in quantile, get the index of the first probability point that is larger than the quantile
        # e.g., if quantile is 0.53, and prob_points are [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], then the index is 5
        indices = torch.searchsorted(self.prob_points, quantile)

        # Get output with shape (batch_size, num-stores, num_quantiles, num__different_lead_times) from forecaster
        x = self.forward(x)
        
        # Get the output corresponding to the lead time
        x = self.retrieve_corresponding_lead_time(x, lead_times)

        # Create the zero-th quantile by taking the difference between the first and second quantile, and subtracting it from the first quantile
        # Do the same for the 1-th quantile
        x = self.create_0_1_quantiles(x) 

        # Get the previous and next quantiles, to use for linear interpolation 
        prev_quantile = torch.gather(x, 2, (indices - 1).unsqueeze(2)).squeeze(2)
        next_quantile = torch.gather(x, 2, indices.unsqueeze(2)).squeeze(2)

        # Get the difference between quantile and the corresponding prob_points
        diff_prev = quantile - self.prob_points[indices - 1]
        diff_next = self.prob_points[indices] - quantile
        sum_diffs = diff_prev + diff_next

        # Get the quantile by using a linear interpolation between the previous and next quantiles
        q = prev_quantile + (next_quantile - prev_quantile)*diff_prev/sum_diffs

        return q
    
    def create_0_1_quantiles(self, x):
        """
        Create the 0-th and 1-th quantiles from the output, by taking the difference between the first and second quantile, 
        and subtracting it from the first quantile
        """

        return torch.cat([(2*x[:, :, 0] - x[:, :, 1]).unsqueeze(2), 
                       x, 
                       (2*x[:, :, -1] - x[:, :, -2]).unsqueeze(2)], 
                       dim=2)
    
    def retrieve_corresponding_lead_time(self, x, lead_times):
        """
        Get the output corresponding to the lead time of each (sample, store)
        """

        # We assume possible lead times are ordered from smallest to largest and are consecutive integers,
        # so we take the difference between the lead time and the minimum one in the list
        lead_times_dif = (lead_times - self.min_lead_time).to(torch.int64)

        return torch.gather(x, 3, lead_times_dif.unsqueeze(2).expand(-1, -1, x.shape[2]).unsqueeze(3)).squeeze(3)
    
    def get_implied_percentile(self, x, lead_time_per_sample, inventory_position, allocation=None, zero_out_no_orders=False):
        """
        Get quantile from predicted distribution
        """
        x = self.forward(x)
        x = self.retrieve_corresponding_lead_time(x, lead_time_per_sample)

        # Create the zero-th quantile by taking the difference between the first and second quantile, and subtracting it from the first quantile
        # Do the same for the 1-th quantile
        x = self.create_0_1_quantiles(x)

        prob_points = torch.tensor([0] + list(self.qs) + [1]).to(self.device).unsqueeze(0)

        # Get index of first quantile that is larger than the inventory position
        indices = torch.clip(torch.searchsorted(x, inventory_position.unsqueeze(2)), min=1, max=prob_points.shape[1] - 1).squeeze(2)

        # Get the probability points corresponding to the previous and next quantile
        prev_percentile = prob_points[torch.zeros_like(indices), torch.clip(indices - 1, min=0)]
        next_percentile = prob_points[torch.zeros_like(indices), torch.clip(indices, max=prob_points.shape[1] - 1)]

        # Get the previous and next quantile
        prev_quantile = torch.gather(x, 2, (indices - 1).unsqueeze(2)).squeeze(2)
        next_quantile = torch.gather(x, 2, indices.unsqueeze(2)).squeeze(2)

        # Get the difference between quantile and the corresponding inventory position
        diff_prev = inventory_position - prev_quantile
        diff_next = next_quantile - inventory_position
        sum_diffs = diff_prev + diff_next

        # Get the percentile by using a linear interpolation between the previous and next quantiles
        percentile = prev_percentile + (next_percentile - prev_percentile)*diff_prev/sum_diffs

        # If zero_out_no_orders is True, then we zero out the implied percentile if the allocation is 0
        if zero_out_no_orders:
            percentile[allocation == 0] = 0

        return percentile
