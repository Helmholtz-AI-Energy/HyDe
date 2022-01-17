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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <torch/extension.h>

/*A method for isotropic TV*/
void rof_iso(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by ,
             torch::Tensor mu, torch::Tensor lambda, int nGS, int nBreg, int width, int height);

/*A method for Anisotropic TV*/
void rof_an(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by ,
            torch::Tensor mu, torch::Tensor lambda, int nGS, int nBreg, int width, int height);

	/*****************Minimization Methods*****************/
void gs_an(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by,
           torch::Tensor mu, torch::Tensor lambda, int width, int height, int iter);
void gs_iso(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by,
            torch::Tensor mu, torch::Tensor lambda, int width, int height, int iter);

	/******************Relaxation Methods*****************/
void gsU(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by,
         torch::Tensor mu, torch::Tensor lambda, int width, int height);
void gsX(torch::Tensor u, torch::Tensor x, torch::Tensor bx , torch::Tensor lambda, int width, int height);
void gsY(torch::Tensor u, torch::Tensor y, torch::Tensor by , torch::Tensor lambda, int width, int height);
void gsSpace(torch::Tensor u, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by, torch::Tensor lambda, int width, int height);
	/************************Bregman***********************/
void bregmanX(torch::Tensor x, torch::Tensor u, torch::Tensor bx, int width, int height);
void bregmanY(torch::Tensor y, torch::Tensor u, torch::Tensor by, int width, int height);

void SplitBregman2(torch::Tensor image, torch::Tensor weight, int max_num_iter, double eps,
                            char isotropic, torch::Tensor out);

/**********************Memory************/

// torch::Tensor newMatrix(int rows, int cols);
// void deleteMatrix(mat ** a);
torch::Tensor copy(torch::Tensor source, torch::Tensor dest, int rows, int cols);

/*********************** sklearn denoising from bregman translated into torch ************/

void SplitBregmanSKImage(torch::Tensor image, torch::Tensor weight, int max_num_iter, float eps,
                            bool isotropic, torch::Tensor out){

    int rows = image.size(0);
    int cols = image.size(1);
    int dims = image.size(2);
    int r, c, k;

    int total = rows * cols * dims;


    torch::Tensor dx = out.clone();
    torch::Tensor dy = out.clone();
    torch::Tensor bx = out.clone();
    torch::Tensor by = out.clone();

    torch::Tensor ux, uy, uprev, unew, bxx, byy, dxx, dyy, s, tx, ty, rmse;
    int i = 0;
    torch::Tensor lam = 2 * weight;
    torch::Tensor norm = (weight + 4 * lam);

    int out_rows, out_cols;

    out_rows = out.size(0);
    out_cols = out.size(1);
    // set the values of out to be the same as the image
    for(r=0; r < rows; r++){
		for(c=0; c < cols; c++){
		    out[r+1][c+1] = image[r][c];
		}
	}
//    # reflect image
//    out[0, 1:out_cols-1] = image[1, :]
//    out[1:out_rows-1, 0] = image[:, 1]
//    out[out_rows-1, 1:out_cols-1] = image[rows-1, :]
//    out[1:out_rows-1, out_cols-1] = image[:, cols-1]

    //    # reflect image
    //    out[0][1:out_cols-1] = image[1][:];
//    out[0][torch::slice(1, out_cols-1)] = image[1][:];
    //    out[out_rows-1][1:out_cols-1] = image[rows-1][:];
    for(c=0; c < cols; c++){
        out[0][c + 1] = image[1][c];
        out[out_rows-1][c + 1] = image[rows - 1][c];
    }
    //    out[1:out_rows-1][0] = image[:][1];
    //    out[1:out_rows-1][out_cols-1] = image[:][cols-1];

    for(r=1; r < rows; r++){
        out[r+1][0] = image[r][1];
        out[r+1][out_cols-1] = image[r][cols-1];
    }

    rmse = lam.zero_();
    rmse += DBL_MAX;

    while ( (i < max_num_iter) && (rmse > eps).item().toBool() ) {
        rmse.zero_();
        for (r=1; r < rows+1; r++){
            for (c=1; c < cols+1; c++){
                for (k=0; k<dims; k++){

                    uprev = out[r][c][k];
                    // forward derivatives
                    ux = out[r][c + 1][k] - uprev;
                    uy = out[r + 1][c][k] - uprev;
                    // Gauss-Seidel method
                    unew = (
                        lam * (
                            out[r + 1][c][k]
                            + out[r - 1][c][k]
                            + out[r][c + 1][k]
                            + out[r][c - 1][k]
                            + dx[r][c - 1][k]
                            - dx[r][c][k]
                            + dy[r - 1][c][k]
                            - dy[r][c][k]

                            - bx[r][c - 1][k]
                            + bx[r][c][k]
                            - by[r - 1][c][k]
                            + by[r][c][k]
                        ) + weight * image[r - 1][c - 1][k]
                    ) / norm;
                    out[r][c][k] = unew;
                    // update root mean square error
                    tx = unew - uprev;
                    rmse += tx * tx;

                    bxx = bx[r][c][k];
                    byy = by[r][c][k];
                    // d_subproblem after reference [4]
                    if (isotropic == 1){
                        tx = ux + bxx;
                        ty = uy + byy;
                        s = torch::sqrt(tx * tx + ty * ty);
                        dxx = s * lam * tx / (s * lam + 1);
                        dyy = s * lam * ty / (s * lam + 1);
                    }
                    else {
                        s = ux + bxx;
                        if ((s > (1 / lam)).item().toBool()){
                            dxx = s - 1/lam;
                        }
                        else if ((s < -1 / lam).item().toBool()){
                            dxx = s + 1 / lam;
                        }
                        else{
                            dxx *= 0;
                        }

                        s = uy + byy;
                        if ((s > 1 / lam).item().toBool()){
                            dyy = s - 1 / lam;
                        }
                        else if ((s < -1 / lam).item().toBool()){
                            dyy = s + 1 / lam;
                        }
                        else{
                            dyy *= 0;
                        }
                    }
                    dx[r][c][k] = dxx;
                    dy[r][c][k] = dyy;

                    bx[r][c][k] += ux - dxx;
                    by[r][c][k] += uy - dyy;
                }
            }
        }
        rmse = torch::sqrt(rmse / total);
        i += 1;
//        std::cout << i << " here" << std::endl;

    }
}




torch::Tensor SplitBregmanROF(torch::Tensor image, torch::Tensor mu, torch::Tensor tol)
{
		/* get the size of the image*/
	// int rows = mxGetN(prhs[0]);
    int rows = image.size(0);
	// int cols = mxGetM(prhs[0]);
    int cols = image.size(1);
//    printf ("rows %d, cols %d", rows, cols);

    /* get the fidelity and convergence parameters*/
	// double mu =  (double)(mxGetScalar(prhs[1]));
	torch::Tensor lambda = 2*mu;
	// double tol = (double)(mxGetScalar(prhs[2]));

	// prhs[0] -> image
	// prhs[1] -> mu
	// prhs[2] -> tol

	/* get the image, and declare memory to hold the auxillary variables*/
	// f is the initial image. apparently it is in the prhs[0]
	/* .data<float>() returns a pointer...but it must be a float */
	auto f = image.clone();
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
//    int i,j;

    /***********Check Conditions******/
	// what do

//    /* Use a copy of the image as an initial guess*/
////    for(i=0;i<rows;i++){
////        for(j=0;j<cols;j++){
////            printf ("i: %d, j: %d\n", i, j);
////            u[i][j] = f[i][j];
////        }
////    }
//    printf ("here2\n");

	/* denoise the image*/

	uOld = torch::zeros({rows, cols});
    count=0;
	do{
		rof_iso(u,f,x,y,bx,by,mu,lambda,1,5,rows,cols);
//        std::cout << count << std::endl;
		diff = copy(u,uOld,rows,cols);
        count++;
//    }while( count<5 );
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
//    printf ("u out: %s", u.toString());
	// TODO: set up return to give what we expect
    return u;
}

/*                IMPLEMENTATION BELOW THIS LINE                         */

/******************Isotropic TV**************/

void rof_iso(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by,
			 torch::Tensor mu, torch::Tensor lambda, int nIter, int nBreg, int width, int height){
    int breg;
    for(breg=0;breg<nBreg;breg++){
        gs_iso(u, f, x, y, bx, by, mu, lambda, width, height, nIter);
//        torch::Tensor** xP;
//        torch::Tensor** uP;
//        torch::Tensor** bxP;
//        torch::Tensor** yP;
//        torch::Tensor** byP;
//        *xP = &x;
//        *uP = &u;
//        *bxP = &bx;
        bregmanX(x, u, bx, width, height);
//        printf ("here\n");
//        *yP = &y;
//        *byP = &by;
        bregmanY(y, u, by, width, height);

    }
}


void gs_iso(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by , torch::Tensor mu, torch::Tensor lambda, int width, int height, int iter){
    int j;
    for(j=0;j<iter;j++){
        gsU(u,f,x,y,bx,by,mu,lambda,width,height);
        gsSpace(u,x,y,bx,by,lambda,width,height);
    }
}

/******************Anisotropic TV**************/
void rof_an(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by ,
            torch::Tensor mu, torch::Tensor lambda, int nGS, int nBreg, int width, int height){
    int breg;
    for(breg=0;breg<nBreg;breg++){
        gs_an(u,f,x,y,bx,by,mu,lambda,width,height,nGS);
//        torch::Tensor** xP;
//        torch::Tensor** uP;
//        torch::Tensor** bxP;
//        torch::Tensor** yP;
//        torch::Tensor** byP;
//        *xP = &x;
//        *uP = &u;
//        *bxP = &bx;
        bregmanX(x, u, bx, width, height);
//        *yP = &y;
//        *byP = &by;
        bregmanY(y, u, by, width, height);
    }
}


void gs_an(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by,
           torch::Tensor mu, torch::Tensor lambda, int width, int height, int iter){
    int j;
    for(j=0;j<iter;j++){
        gsU(u,f,x,y,bx,by,mu,lambda,width,height);
//        torch::Tensor** xP;
//        torch::Tensor** uP;
//        torch::Tensor** bxP;
//        torch::Tensor** yP;
//        torch::Tensor** byP;
//        *xP = &x;
//        *uP = &u;
//        *bxP = &bx;
        gsX(u, x, bx, lambda, width, height);
//        *yP = &y;
//        *byP = &by;
        gsY(u, y, by, lambda, width, height);
    }
}

/****Relaxation operators****/

void gsU(torch::Tensor u, torch::Tensor f, torch::Tensor x, torch::Tensor y, torch::Tensor bx,
         torch::Tensor by , torch::Tensor mu, torch::Tensor lambda, int width, int height){
	int w,h;
	torch::Tensor sum;
	int wm1,hm1;
	torch::Tensor normConst = 1.0/(mu+4*lambda);
	int wSent = width-1, hSent = height-1;
	for(w=1;w<wSent;w++){		/* do the central pixels*/
		wm1 = w-1;
		for(h=1;h<hSent;h++){
			hm1 = h-1;
			sum = x[wm1][h] - x[w][h]+y[w][hm1] - y[w][h] - bx[wm1][h] + bx[w][h]-by[w][hm1] + by[w][h];
//            printf ("here2\n");
			sum += u[w+1][h] + u[wm1][h] + u[w][h+1] + u[w][hm1];
//			printf ("here3\n");
			sum *= lambda;
//			printf ("here4\n");
            // this is the line that is failing
//            std::cout << "sum " << sum << std::endl;
//            std::cout << "mu " << mu << std::endl;
//            std::cout << "f " << f.sizes() << std::endl;
			sum += mu * f[w][h];
//			printf ("here5\n");
			sum *= normConst;
//			printf ("here6\n");
			u[w][h] = sum;
		}
	}
    w=0;				/* do the left pixels*/
    for(h=1;h<hSent;h++){
        sum = - x[w][h]+y[w][h-1] - y[w][h] + bx[w][h]-by[w][h-1] + by[w][h];
        sum+=(u[w+1][h]+u[w][h+1]+u[w][h-1]);
        sum*=lambda;
        sum+=mu*f[w][h];
        sum/=mu+3*lambda;
        u[w][h] = sum;
    }
    w = width-1;		/* do the right pixels*/
    for(h=1;h<hSent;h++){
        sum = x[w-1][h] +y[w][h-1] - y[w][h] -bx[w-1][h] -by[w][h-1] + by[w][h];
        sum+=u[w-1][h]+u[w][h+1]+u[w][h-1];
        sum*=lambda;
        sum+=mu*f[w][h];
        sum/=mu+3*lambda;
        u[w][h] = sum;
    }
//    printf ("here3\n");
    h=0;
    for(w=1;w<wSent;w++){		/* do the top pixels*/
        sum = x[w-1][h] - x[w][h] - y[w][h] -bx[w-1][h] + bx[w][h] + by[w][h];
        sum+=u[w+1][h]+u[w-1][h]+u[w][h+1];
        sum*=lambda;
        sum+=mu*f[w][h];
        sum/=mu+3*lambda;
        u[w][h] = sum;
    }
    h = height-1;
    for(w=1;w<wSent;w++){		/* do the bottom pixels*/
        sum = x[w-1][h] - x[w][h]+y[w][h-1] -bx[w-1][h] + bx[w][h]-by[w][h-1];
        sum+=u[w+1][h]+u[w-1][h]+u[w][h-1];
        sum*=lambda;
        sum+=mu*f[w][h];
        sum/=mu+3*lambda;
        u[w][h] = sum;
    }
    /* do the top left pixel*/
    w=h=0;
    sum =  - x[w][h] - y[w][h] + bx[w][h] + by[w][h];
    sum+=u[w+1][h]+u[w][h+1];
    sum*=lambda;
    sum+=mu*f[w][h];
    sum/=mu+2*lambda;
    u[w][h] = sum;
    /* do the top right pixel*/
    w=width-1; h=0;
    sum = x[w-1][h]  - y[w][h] - bx[w-1][h] + by[w][h];
    sum+=u[w-1][h]+u[w][h+1];
    sum*=lambda;
    sum+=mu*f[w][h];
    sum/=mu+2*lambda;
    u[w][h] = sum;
    /* do the bottom left pixel*/
    w=0; h=height-1;
    sum =  - x[w][h]+y[w][h-1] + bx[w][h]-by[w][h-1] ;
    sum+=u[w+1][h]+u[w][h-1];
    sum*=lambda;
    sum+=mu*f[w][h];
    sum/=mu+2*lambda;
    u[w][h] = sum;
    /* do the bottom right pixel*/
    w=width-1; h=height-1;
    sum = x[w-1][h]+y[w][h-1] -bx[w-1][h] -by[w][h-1];
    sum+=u[w-1][h]+u[w][h-1];
    sum*=lambda;
    sum+=mu*f[w][h];
    sum/=mu+2*lambda;
    u[w][h] = sum;
}


void gsSpace(torch::Tensor u, torch::Tensor x, torch::Tensor y, torch::Tensor bx, torch::Tensor by,
             torch::Tensor lambda, int width, int height){
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


void gsX(torch::Tensor u, torch::Tensor x, torch::Tensor bx , torch::Tensor lambda, int width, int height){
	int w,h;
	torch::Tensor base;
	const torch::Tensor flux = 1.0/lambda;
	const torch::Tensor mflux = -1.0/lambda;
	torch::Tensor uwp1;
	torch::Tensor uw;
	torch::Tensor bxw;
	torch::Tensor xw;
    width = width-1;
	for(w=0;w<width;w++){
		uwp1 = u[w+1];
		uw = u[w];
		bxw = bx[w];
		xw = x[w];
		for(h=0;h<height;h++){
			base = (uwp1[h]-uw[h]+bxw[h]);
			if((base>flux).item().toBool()) {
				xw[h] = base-flux;
				continue;
			}
			if((base<mflux).item().toBool()){
				xw[h] = base+flux;
				continue;
			}
			xw[h] *= 0;
		}
	}
}

void gsY(torch::Tensor u, torch::Tensor y, torch::Tensor by , torch::Tensor lambda, int width, int height){
	int w,h;
	torch::Tensor base;
	const torch::Tensor flux = 1.0/lambda;
	const torch::Tensor mflux = -1.0/lambda;
	torch::Tensor uw;
	torch::Tensor yw;
	torch::Tensor bw;
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

void bregmanX(torch::Tensor x, torch::Tensor u, torch::Tensor bx, int width, int height){
    int w,h;
    torch::Tensor d;
    torch::Tensor uwp1,uw,bxw,xw;
    for(w=0;w<width-1;w++){
        uwp1=u[w+1];uw=u[w];bxw=bx[w];xw=x[w];
        for(h=0;h<height;h++){
            d = uwp1[h]-uw[h];
            bxw[h]+= d-xw[h];
        }
    }
}


void bregmanY(torch::Tensor y,torch::Tensor u, torch::Tensor by, int width, int height){
    int w,h;
    torch::Tensor d;
    int hSent = height-1;
    torch::Tensor uw,byw,yw;
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
	torch::Tensor temp, sumDiff, sum;
//	for(r=0;r<rows;r++)
//		for(c=0;c<cols;c++){
//			temp = dest[r][c];
//			sum+=temp*temp;
//			temp -= source[r][c];
//			sumDiff +=temp*temp;
//			dest[r][c]=source[r][c];
//		}
//    sum2 = dest.pow(2).sum()
    sum = torch::pow(dest, 2).sum();
    sumDiff = (dest - source).pow(2).sum();
    dest = source.clone();
	return torch::sqrt(sumDiff/sum);
}


//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("split_bregman_rof", &SplitBregmanROF, "SplitBregmanROF");
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("split_bregman_skimage", &SplitBregmanSKImage, "SplitBregmanSKImage");
}
