/*				SplitBregmanROF.c (MEX version) by Tom Goldstein
 *   This code performs isotropic ROF denoising using the "Split Bregman" algorithm.
 * This version of the code has a "mex" interface, and should be compiled and called
 * through MATLAB.
 *
 *DISCLAIMER:  This code is for academic (non-commercial) use only.  Also, this code
 *comes with absolutely NO warranty of any kind: I do my best to write reliable codes,
 *but I take no responsibility for the reliability of the results.
 *
 *                      HOW TO COMPILE THIS CODE
 *   To compile this code, open a MATLAB terminal, and change the current directory to
 *the folder where this "c" file is contained.  Then, enter this command:
 *    >>  mex splitBregmanROF.c
 *This file has been tested under windows using lcc, and under SUSE Unix using gcc.
 *Once the file is compiled, the command "splitBregmanROF" can be used just like any
 *other MATLAB m-file.
 *
 *                      HOW TO USE THIS CODE
 * An image is denoised using the following command
 *
 *   SplitBregmanROF(image,mu,tol);
 *
 * where:
 *   - "image" is a 2d array containing the noisy image.
 *   - "mu" is the weighting parameter for the fidelity term
 *            (usually a value between 0.01 and 0.5 works well for images with
 *                           pixels on the 0-255 scale).
 *   - "tol" is the stopping tolerance for the iteration.  "tol"=0.001 is reasonable for
 *            most applications.
 *
 */

#include <math.h>
// #include "mex.h"
#include <stdio.h>
#include <stdlib.h>

#include <torch/extension.h>
typedef torch::Tensor mat;


/*A method for isotropic TV*/
void rof_iso(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by ,
					 double mu, double lambda, int nGS, int nBreg, int width, int height);

/*A method for Anisotropic TV*/
void rof_an(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by ,
					 double mu, double lambda, int nGS, int nBreg, int width, int height);

	/*****************Minimization Methods*****************/
void gs_an(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by , double mu, double lambda, int width, int height, int iter);
void gs_iso(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by , double mu, double lambda, int width, int height, int iter);

	/******************Relaxation Methods*****************/
void gsU(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by , double mu, double lambda, int width, int height);
void gsX(torch::Tensor u, torch::Tensor x, torch::Tensor bx , double lambda, int width, int height);
void gsY(torch::Tensor u, torch::Tensor y, torch::Tensor by , double lambda, int width, int height);
void gsSpace(torch::Tensor u, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by, double lambda, int width, int height);

	/************************Bregman***********************/
void bregmanX(torch::Tensor x,torch::Tensor u, torch::Tensor bx, int width, int height);
void bregmanY(torch::Tensor y,torch::Tensor u, torch::Tensor by, int width, int height);

/**********************Memory************/

// torch::Tensor newMatrix(int rows, int cols);
// void deleteMatrix(mat ** a);
torch::Tensor copy(torch::Tensor source, torch::Tensor dest, int rows, int cols);

/***********************The MEX Interface to rof_iso()***************/

torch::Tensor SplitBregmanROF(torch::Tensor image, double mu, double tol)
{
		/* get the size of the image*/
	// int rows = mxGetN(prhs[0]);
    int rows = image.size(0);
	// int cols = mxGetM(prhs[0]);
    int cols = image.size(1);

		/* get the fidelity and convergence parameters*/
	// double mu =  (double)(mxGetScalar(prhs[1]));
	double lambda = 2*mu;
	// double tol = (double)(mxGetScalar(prhs[2]));

	// prhs[0] -> image
	// prhs[1] -> mu
	// prhs[2] -> tol

	/* get the image, and declare memory to hold the auxillary variables*/
	// f is the initial image. apparently it is in the prhs[0]
	/* .data<float>() returns a pointer...but it must be a float */
	auto f = image;
	auto u = torch::zeros({rows, cols});
	auto x = torch::zeros({rows-1,cols});
	auto y = torch::zeros({rows,cols-1});
	auto bx = torch::zeros({rows-1,cols});
	auto by = torch::zeros({rows,cols-1});

    torch::Tensor uOld;
    // torch::Tensor *outArray;
    torch::Tensor diff;

	// torch::Tensor tol = torch::tensor(tol);
    int count;
    int i,j;

    /***********Check Conditions******/
	// what do

    /* Use a copy of the image as an initial guess*/
		for(i=0;i<rows;i++){
		    for(j=0;j<cols;j++){
		        u[i][j] = f[i][j];
		    }
		}

	/* denoise the image*/

	uOld = torch::zeros({rows, cols});
    count=0;
	do{
		rof_iso(u,f,x,y,bx,by,mu,lambda,1,5,rows,cols);
		diff = copy(u,uOld,rows,cols);
        count++;
	}while( ((diff>tol).item().toBool() && count<1000) || count<5 );

	/* copy denoised image to output vector*/
	// plhs[0] = mxCreateDoubleMatrix(cols, rows, mxREAL); /*mxReal is our data-type*/
	// outArray = mxGetPr(plhs[0]);

	// for(i=0;i<rows;i++){
	//     for(j=0;j<cols;j++){
	//         outArray[(i*cols)+j] = u[i][j];
	//     }
	// }

	/* Free memory */
	// is this needed still?
	// delete u;
	// delete x;
	// delete bx;
	// delete by;
	// delete uOld;

	// TODO: set up return to give what we expect
    return u;
}





/*                IMPLEMENTATION BELOW THIS LINE                         */

/******************Isotropic TV**************/

void rof_iso(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by,
			 double mu, double lambda, int nGS, int nBreg, int width, int height){
		int breg;
		for(breg=0;breg<nBreg;breg++){
				gs_iso(u,f,x,y,bx,by,mu,lambda,width, height,nGS);
				bregmanX(x,u,bx,width,height);
				bregmanY(y,u,by,width,height);
		}
}


void gs_iso(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by , double mu, double lambda, int width, int height, int iter){
		int j;
		for(j=0;j<iter;j++){
			gsU(u,f,x,y,bx,by,mu,lambda,width,height);
			gsSpace(u,x,y,bx,by,lambda,width,height);

		}
}


/******************Anisotropic TV**************/
void rof_an(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by ,
								 double mu, double lambda, int nGS, int nBreg, int width, int height){
		int breg;
		for(breg=0;breg<nBreg;breg++){
				gs_an(u,f,x,y,bx,by,mu,lambda,width, height,nGS);
				bregmanX(x,u,bx,width,height);
				bregmanY(y,u,by,width,height);
		}
}


void gs_an(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by , double mu, double lambda, int width, int height, int iter){
		int j;
		for(j=0;j<iter;j++){
			gsU(u,f,x,y,bx,by,mu,lambda,width,height);
			gsX(u,x,bx,lambda,width,height);
			gsY(u,y,by,lambda,width,height);
		}
}





/****Relaxation operators****/

void gsU(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by , double mu, double lambda, int width, int height){
	int w,h;
	torch::Tensor sum;
	int wm1,hm1;
	double normConst = 1.0/(mu+4*lambda);
	int wSent = width-1, hSent = height-1;
	for(w=1;w<wSent;w++){		/* do the central pixels*/
		wm1 = w-1;
		for(h=1;h<hSent;h++){
			hm1 = h-1;
			sum = x[wm1][h] - x[w][h]+y[w][hm1] - y[w][h]
						-bx[wm1][h] + bx[w][h]-by[w][hm1] + by[w][h];
			sum+=(u[w+1][h]+u[wm1][h]+u[w][h+1]+u[w][hm1]);
			sum*=lambda;
			sum+=mu*f[w][h];
			sum*=normConst;
			u[w][h] = sum;
		}
	}
		w=0;				/* do the left pixels*/
		for(h=1;h<hSent;h++){
			sum = - x[w][h]+y[w][h-1] - y[w][h]
						+ bx[w][h]-by[w][h-1] + by[w][h];
			sum+=(u[w+1][h]+u[w][h+1]+u[w][h-1]);
			sum*=lambda;
			sum+=mu*f[w][h];
			sum/=mu+3*lambda;
			u[w][h] = sum;
		}
		w = width-1;		/* do the right pixels*/
			for(h=1;h<hSent;h++){
				sum = x[w-1][h] +y[w][h-1] - y[w][h]
								-bx[w-1][h] -by[w][h-1] + by[w][h];
				sum+=u[w-1][h]+u[w][h+1]+u[w][h-1];
				sum*=lambda;
				sum+=mu*f[w][h];
				sum/=mu+3*lambda;
				u[w][h] = sum;
			}

		h=0;
		for(w=1;w<wSent;w++){		/* do the top pixels*/
				sum = x[w-1][h] - x[w][h] - y[w][h]
								-bx[w-1][h] + bx[w][h] + by[w][h];
				sum+=u[w+1][h]+u[w-1][h]+u[w][h+1];
				sum*=lambda;
				sum+=mu*f[w][h];
				sum/=mu+3*lambda;
				u[w][h] = sum;
			}
		h = height-1;
		for(w=1;w<wSent;w++){		/* do the bottom pixels*/
				sum = x[w-1][h] - x[w][h]+y[w][h-1]
								-bx[w-1][h] + bx[w][h]-by[w][h-1];
				sum+=u[w+1][h]+u[w-1][h]+u[w][h-1];
				sum*=lambda;
				sum+=mu*f[w][h];
				sum/=mu+3*lambda;
				u[w][h] = sum;
			}
			/* do the top left pixel*/
			w=h=0;
				sum =  - x[w][h] - y[w][h]
								+ bx[w][h] + by[w][h];
				sum+=u[w+1][h]+u[w][h+1];
				sum*=lambda;
				sum+=mu*f[w][h];
				sum/=mu+2*lambda;
				u[w][h] = sum;
				/* do the top right pixel*/
			w=width-1; h=0;
				sum = x[w-1][h]  - y[w][h]
								-bx[w-1][h] + by[w][h];
				sum+=u[w-1][h]+u[w][h+1];
				sum*=lambda;
				sum+=mu*f[w][h];
				sum/=mu+2*lambda;
				u[w][h] = sum;
				/* do the bottom left pixel*/
			w=0; h=height-1;
				sum =  - x[w][h]+y[w][h-1]
								+ bx[w][h]-by[w][h-1] ;

				sum+=u[w+1][h]+u[w][h-1];
				sum*=lambda;
				sum+=mu*f[w][h];
				sum/=mu+2*lambda;
				u[w][h] = sum;
				/* do the bottom right pixel*/
			w=width-1; h=height-1;
				sum = x[w-1][h]+y[w][h-1]
								-bx[w-1][h] -by[w][h-1];
				sum+=u[w-1][h]+u[w][h-1];
				sum*=lambda;
				sum+=mu*f[w][h];
				sum/=mu+2*lambda;
				u[w][h] = sum;
}


void gsSpace(torch::Tensor u, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by, torch::Tensor lambda, int width, int height){
	int w,h;
	torch::Tensor a,b,s;
	torch::Tensor flux = 1.0/lambda;
	torch::Tensor mflux = -1.0/lambda;
	torch::Tensor flux2 = flux*flux;
	torch::Tensor uw,uwp1,bxw,byw,xw,yw;
    torch::Tensor base;
	for(w=0;w<width-1;w++){
		uw=u[w];
		uwp1=u[w+1];
		bxw=bx[w];
		byw=by[w];
		xw=x[w];
		yw=y[w];
		for(h=0;h<height-1;h++){
			a =  uwp1[h]-uw[h]+bxw[h];
			b =  uw[h+1]-uw[h]+byw[h];
			s = a*a+b*b;
			if((s<flux2).item().toBool()){
				xw[h]=0;
				yw[h]=0;
				continue;
			}
			s = sqrt(s);
			s=(s-flux)/s;
			xw[h] = s*a;
			yw[h] = s*b;
		}
	}

	h = height-1;
	for(w=0;w<width-1;w++){
			base =  u[w+1][h]-u[w][h]+bx[w][h];
			if ((base>flux).item().toBool()) {x[w][h] = base-flux; continue;}
			if((base<mflux).item().toBool()) {x[w][h] = base+flux; continue;}
			x[w][h] = 0;
	}
	w = width-1;
	for(h=0;h<height-1;h++){
		base =  u[w][h+1]-u[w][h]+by[w][h];
		if((base>flux).item().toBool()) {y[w][h] = base-flux; continue;}
		if((base<mflux).item().toBool()) {y[w][h] = base+flux; continue;}
		y[w][h] = 0;
	}
}


void gsX(torch::Tensor** u, torch::Tensor** x, torch::Tensor** bx , double lambda, int width, int height){
	int w,h;
	double base;
	const double flux = 1.0/lambda;
	const double mflux = -1.0/lambda;
	torch::Tensor* uwp1;
	torch::Tensor* uw;
	torch::Tensor* bxw;
	torch::Tensor* xw;
    width = width-1;
	for(w=0;w<width;w++){
		uwp1 = u[w+1];
		uw = u[w];
		bxw = bx[w];
		xw = x[w];
		for(h=0;h<height;h++){
			base = (uwp1[h]-uw[h]+bxw[h]).item().toDouble();
			if(base>flux) {
				xw[h] = torch::tensor(base-flux);
				continue;
			}
			if(base<mflux){
				xw[h] = torch::tensor(base+flux);
				continue;
			}
			xw[h] *= 0;
		}
	}
}

void gsY(torch::Tensor** u, torch::Tensor** y, torch::Tensor** by , double lambda, int width, int height){
	int w,h;
	torch::Tensor base;
	const double flux = 1.0/lambda;
	const double mflux = -1.0/lambda;
	torch::Tensor* uw;
	torch::Tensor* yw;
	torch::Tensor* bw;
    height = height-1;
	for(w=0;w<width;w++){
		uw = u[w];
		yw = y[w];
		bw = by[w];
		for(h=0;h<height;h++){
			base = uw[h+1]-uw[h]+bw[h];
			if((base>flux).item().toBool()) {yw[h] = base-flux; continue;}
			if((base<mflux).item().toBool()){yw[h] = base+flux; continue;}
			yw[h] = torch::tensor(0);
		}
	}
}

void bregmanX(torch::Tensor** x,torch::Tensor** u, torch::Tensor** bx, int width, int height){
		int w,h;
		torch::Tensor d;
		torch::Tensor* uwp1,*uw,*bxw,*xw;
		for(w=0;w<width-1;w++){
			uwp1=u[w+1];uw=u[w];bxw=bx[w];xw=x[w];
			for(h=0;h<height;h++){
				d = uwp1[h]-uw[h];
				bxw[h]+= d-xw[h];
			}
		}
	}


void bregmanY(torch::Tensor** y,torch::Tensor** u, torch::Tensor** by, int width, int height){
		int w,h;
		torch::Tensor d;
		int hSent = height-1;
		torch::Tensor* uw,*byw,*yw;
		for(w=0;w<width;w++){
			uw=u[w];byw=by[w];yw=y[w];
			for(h=0;h<hSent;h++){
				d = uw[h+1]-uw[h];
				byw[h]+= d-yw[h];
			}
		}
	}

/************************memory****************/

torch::Tensor copy(torch::Tensor source, torch::Tensor dest, int rows, int cols){
	int r,c;
	torch::Tensor temp,sumDiff, sum;
	for(r=0;r<rows;r++)
		for(c=0;c<cols;c++){
			temp = dest[r][c];
			sum+=temp*temp;
			temp -= source[r][c];
			sumDiff +=temp*temp;

			dest[r][c]=source[r][c];

		}
	return torch::sqrt(sumDiff/sum);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &SplitBregmanROF, "SplitBregmanROF");
}
