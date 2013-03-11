classdef AIS < handle
%only support binary RBM now    
    
% Code provided by Ruslan Salakhutdinov
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
  
    properties
        beta = [0:1/1000:0.5 0.5:1/10000:0.9 0.9:1/100000:1.0];
        numruns = 100;
    end
    
    methods
        function self = AIS()
        end
        
        function [logZZ_est, logZZ_est_up, logZZ_est_down] = run(self,learner, X) %only support 1 batch for now 
            [feadim numdata]=size(X);
             count_int = sum(X,2);
             
             lp=5;
             p_int = (count_int+lp)/(numdata+lp);
             log_base_rate = log( p_int) - log(1-p_int);
             visbiases_base = log_base_rate';
                
             visbias_base = repmat(visbiases_base,self.numruns,1); %biases of base-rate model.  
             hidbias = repmat(learner.hbias',self.numruns,1); 
             visbias = repmat(learner.vbias',self.numruns,1);

             %%%% Sample from the base-rate model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             logww = zeros(self.numruns,1);
             negdata = repmat(1./(1+exp(-visbiases_base)),self.numruns,1);  
             negdata = negdata > rand(self.numruns,feadim);
             logww  =  logww - (negdata*visbiases_base' + learner.numunits*log(2));

             Wh = negdata*learner.weights + hidbias; 
             Bv_base = negdata*visbiases_base';
             Bv = negdata*learner.vbias;   
             tt=1; 
             
            for bb = self.beta(2:end-1);  
%                fprintf(1,'beta=%d\r',bb);
               tt = tt+1; 

               expWh = exp(bb*Wh);
               logww  =  logww + (1-bb)*Bv_base + bb*Bv + sum(log(1+expWh),2);

               poshidprobs = expWh./(1 + expWh);
               poshidstates = poshidprobs > rand(self.numruns,learner.numunits);

               negdata = 1./(1 + exp(-(1-bb)*visbias_base - bb*(poshidstates*learner.weights' + visbias)));
               negdata = negdata > rand(self.numruns,feadim);

               Wh      = negdata*learner.weights + hidbias;
               Bv_base = negdata*visbiases_base';
               Bv      = negdata*learner.vbias;

               expWh = exp(bb*Wh);
               logww  =  logww - ((1-bb)*Bv_base + bb*Bv + sum(log(1+expWh),2));
            end 
             
             expWh = exp(Wh);
             logww  = logww +  negdata*learner.vbias + sum(log(1+expWh),2);

             %%% Compute an estimate of logZZ_est +/- 3 standard deviations.   
             r_AIS = Utils.logfunc(logww(:),@sum) -  log(self.numruns);  
             aa = mean(logww(:)); 
             logstd_AIS = log (std(exp ( logww-aa))) + aa - log(self.numruns)/2;   
             %%% Same as computing  logstd_AIS = log(std(exp(logww(:)))/sqrt(numcases));  

             logZZ_base = sum(log(1+exp(visbiases_base))) + (learner.numunits)*log(2); 
             logZZ_est = r_AIS + logZZ_base;
             logZZ_est_up = Utils.logfunc([log(3)+logstd_AIS;r_AIS],@sum) + logZZ_base;
             logZZ_est_down = Utils.logfunc([(log(3)+logstd_AIS);r_AIS],@diff) + logZZ_base;
        end
    end
    
end

