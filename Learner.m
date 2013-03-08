classdef Learner < handle
    properties
        preprocessor; %for preprocessing, preprocessor.run is the function handle
        
        %training variables (should be moved to Optimizer in the future)
        max_iter = 100;
        save_iter;
        save_dir;
    end
    
    methods
        function self = Learner()            
        end
        

        
        function X = fprop(self, X) %in current design train need to preprocess manually because other trainer (Optimizer) might envolve
            if ~isempty(self.preprocessor)
                X = self.preprocessor.run(self.preprocessor, X);     
            end
        end
        
        function train(self,X) %train with all data
            self.initialization(X);
            self.initIter(1);
            self.update(X);            
            self.save();
        end
        
        %-----to support batch update
        function [] = initialization(self, X, batch_size)            
        end
        
        function [] = initIter(self,t)            
        end
        
        function [] = update(self, X)                                    
        end
        
        function [isstop] = checkStop(self)        
            isstop = false;
        end
        
        function [] = save(self)
        end
    end
end