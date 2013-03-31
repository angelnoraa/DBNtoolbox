classdef RBMSM < handle & RBM
    %rbm updated by score matching
    %only gaussian input is implemented
    properties    
%         sigma_epoch; %gradually change sigma
        opt;      
        gs; %GroupSparsity object
        numParam;
        
        X; %make a copy of X to update sigma
    end
    
    methods
        function self = RBMSM(numunits)                                    
            self = self@RBM(numunits, 'gau');                  
            self.opt = Optimizer();
            self.sigma = [];            
        end                
        
        function self = initFromGMM(self,gmm)
            %gmm is Matlab stat toolbox object
            self.vbias = gmm.mu(1,:)';
            self.sigma = sqrt(gmm.Sigma');
            self.weights = bsxfun(@rdivide,bsxfun(@minus,gmm.mu(2:end,:)',self.vbias),self.sigma);
            self.hbias = log(gmm.PComponents(2:end)/gmm.PComponents(1))' - 0.5*sum(self.weights.^2,1)' ...
                - bsxfun(@rdivide, self.weights, self.sigma)'*self.vbias;
        end                
        
        function [] = initialization(self, X, batch_size)                 
            self.initialization@RBM(X,batch_size);
            if isempty(self.sigma)
                self.sigma = ones(self.feadim,1);
            end                   
            self.numParam = [numel(self.weights), numel(self.hbias), numel(self.vbias)];
            self.numParam = cumsum(self.numParam);
            self.X = X;
            
            if ~isempty(self.gs)
                self.gs.setPar(self.numunits);
                self.gs.numdata = batch_size;
            end
        end
       
        function [] = initIter(self,t)            
            self.initIter@RBM(t);
%             self.sigma_epoch = [];            
            if t == 1, return; end
            if mod(t,5) == 1 %only update this for a long time
                self.sigma = self.opt.run(@(paramvec) self.fobjSigma(paramvec, self.X), self.sigma); 
            end
        end
        
        function [] = update(self, X)			
            theta = self.vectorizeParam();      
            theta = self.opt.run(@(paramvec) self.fobj(paramvec, X), theta); 
            self.devectorizeParam(theta);      
        end
		
        function [f g] = fobj(self, paramvec, X)        
            %times sigma square 
            self.devectorizeParam(paramvec);            
                        
            W = self.weights;    
            Ws = bsxfun(@times, W, self.sigma);
            Vs = bsxfun(@rdivide, X , self.sigma);
            h = Utils.sigmoid(W'*Vs+ self.hbias*ones(1,size(X,2)));            

            vmc = bsxfun(@minus,X,self.vbias);
            recon_err = vmc - Ws*h;
            ha = h.*(1-h);
            
            f = mean(sum(0.5*(recon_err.^2),1),2);    
                        
            WsReconHa = Ws'*recon_err .* ha;
            Vss = bsxfun(@rdivide, Vs, self.sigma);
            dW = -bsxfun(@times,recon_err,self.sigma)*h' - Vs*WsReconHa';
            dh = -mean(WsReconHa,2);
            dv = -mean(recon_err,2);
%             ds = -mean(recon_err.*(W*h),2) + mean((W*WsReconHa).* Vss,2);
                       
            f = f + mean(sum((W.^2)*ha,1),2);
            Wsum = sum(W.^2,1)';
            WsumHa1m2h = bsxfun(@times,(1-2*h).*ha,Wsum);
            dW = dW + 2*bsxfun(@times, W, sum(ha,2)') + Vs*WsumHa1m2h';
            dh = dh + mean(WsumHa1m2h,2);
%             ds = ds - mean((W*WsumHa1m2h) .* Vss,2);
                        
            %add sparsity gradient
            if ~isempty(self.gs)
                self.gs.IN = h;
                self.gs.fprop();
                [fspars, dspars] = self.gs.bprop(0,0);
                f = f + fspars/size(X,2);
                dsha = dspars.*ha;
                dW = dW + Vs*dsha';
                dh = dh + mean(dsha,2);                
            end
            
            dW = dW / size(X,2);            
            
            f = f + self.l2_C*0.5*sum(Utils.vec(W.^2));
            dW = dW + self.l2_C*W;
            
            g = [dW(:) ; dh; dv] ;
            
            self.recon_err_epoch = [self.recon_err_epoch f]; %note that this is not recon_err
            self.sparsity_epoch = [self.sparsity_epoch mean(h(:))];            
        end
        
       function [f g] = fobjSigma(self, paramvec, X)        
            self.sigma = paramvec;            
                        
            W = self.weights;    
            Ws = bsxfun(@times, W, self.sigma);
            Vs = bsxfun(@rdivide, X , self.sigma);
            h = Utils.sigmoid(W'*Vs+ self.hbias*ones(1,size(X,2)));            

            vmc = bsxfun(@minus,X,self.vbias);
            recon_err = vmc - Ws*h;
            ha = h.*(1-h);
            
            f = mean(sum(0.5*(recon_err.^2),1),2);    
                        
            WsReconHa = Ws'*recon_err .* ha;
            Vss = bsxfun(@rdivide, Vs, self.sigma);            
            ds = -mean(recon_err.*(W*h),2) + mean((W*WsReconHa).* Vss,2);
                       
            f = f + mean(sum((W.^2)*ha,1),2);
            Wsum = sum(W.^2,1)';
            WsumHa1m2h = bsxfun(@times,(1-2*h).*ha,Wsum);            
            ds = ds - mean((W*WsumHa1m2h) .* Vss,2);
            
                        %add sparsity gradient
            if ~isempty(self.gs)
                self.gs.IN = h;
                self.gs.fprop();
                [fspars, dspars] = self.gs.bprop(0,0);
                f = f + fspars/size(X,2);
                dsha = dspars.*ha;
                ds = ds - mean((W*dsha).*Vss,2);
            end                                    
            
            g = [ds] ;            
        end
        
        function [isstop] = checkStop(self)                   
            isstop = self.checkStop@RBM();
%             self.sigma = sqrt(mean(self.sigma_epoch,2));
        end
		
        function [acti, states] = fprop(self, X, constraint)          
            X = fprop@Learner(self,X);
            Wv = self.weights'*bsxfun(@rdivide,X,self.sigma)+ self.hbias*ones(1,size(X,2));
            if ~exist('constraint','var') || constraint == false
                acti = Utils.sigmoid(Wv);  							  			
                if nargout > 1
                    states = acti > rand(size(acti));
                end
            else
                Wv = [Wv; zeros(1, size(X,2))];
                Wv = exp(bsxfun(@minus, Wv, max(Wv, [], 1))); %buffer                                
                acti = bsxfun(@rdivide, Wv, sum(Wv,1));      
                if nargout > 1 
                    states = mnrnd(1,acti')';
                    states = states(1:end-1,:);
                end
                acti = acti(1:end-1,:);
            end
        end                        
        function theta = vectorizeParam(self)
            theta = [self.weights(:) ; self.hbias; self.vbias];
        end
        
        function [] = devectorizeParam(self, paramvec)
            self.weights = reshape(paramvec(1:self.numParam(1)),size(self.weights));
            self.hbias = paramvec(self.numParam(1)+1:self.numParam(2));
            self.vbias = paramvec(self.numParam(2)+1:self.numParam(3));
%             self.sigma = paramvec(self.numParam(3)+1:self.numParam(4)); %for stabalization?
        end                      
    end
    
    methods(Static)
        function [] = checkGradient()
            learner = RBMSM(10);
            gs = GroupSparsity(1,1);
            gs.target_sparsity = 0.1;
            gs.lambda = 5;
            learner.gs = gs;
            
            X = randn(8,3);
            learner.initialization(X, 3);    
            learner.l2_C = 1;
            [d dbp dnum] = Utils.checkgrad2_nodisplay(@(paramvec) learner.fobj(paramvec, X), learner.vectorizeParam, 1e-3);
            d
            
            [d dbp dnum] = Utils.checkgrad2_nodisplay(@(paramvec) learner.fobjSigma(paramvec, X), learner.sigma, 1e-3);
            d
        end
        
        function [] = testGMM2RBM(rbm,gmm,x)
            %x = feadim*1
%             tmp = bsxfun(@plus,-gmm.mu',x);
%             bsxfun(@rdivide, tmp, gmm.Sigma'
%             gmm.PComponents 
        end
    end
      
end