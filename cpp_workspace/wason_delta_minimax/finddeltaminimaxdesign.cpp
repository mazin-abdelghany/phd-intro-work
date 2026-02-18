#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // technically should be changed to cmath
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cstdlib> 
#include <vector>
#include <time.h>

using namespace std;

void printvector(vector<double>& vector)
{
    size_t i;

    for(i = 0; i < vector.size(); i++)
    {
        std::cout << vector.at(i) << " ";
    }
    
    std::cout << "\n";
}

// Wichmann–Hill uniform pseudorandom number generator
// Wichmann, Brian A.; Hill, I. David (1982). "Algorithm AS 183: An Efficient 
// and Portable Pseudo-Random Number Generator". Journal of the Royal 
// Statistical Society. Series C (Applied Statistics). 
double asran(void)
{
    static long ix=1,iy=1, iz=1;

    double r{};
  
    iz = iz;
    ix = (171*ix) % 30269;
    iy = (172*iy) % 30307;
    iz = (170*iz) % 30323;
    r  = (double)ix/30269.0 + (double)iy/30307.0 + (double)iz/30323.0;
    return ( r - (int) r );
}

// finds normal pdf, cdf and inverse cdf
double normalpdf(double z)
{
    return ((1.)/sqrt(2*M_PI)) * exp((-z*z)/2);
}

// Abramowitz & Stegun (1964) approximation 
// of the standard normal cdf accurate to < 7.5e-8 based on
// Abramowitz/Stegun 26.2.17
// maybe originally in  Approximations for Digital Computers
// Cecil Hastings 1955
double normalcdf(double z)
{
    if (z > 6.0)
    {
        return 1.0;
    }
  
    if (z < -6.0)
    {
        return 0.0;
    }

    double b1 = 0.31938153;
    double b2 = -0.356563782;
    double b3 = 1.781477937;
    double b4 = -1.821255978;
    double b5 = 1.330274429;

    double p = 0.2316419;

    // numerical approximation for 1/sqrt(2*pi)
    double c2 = 0.3989423;

    double a = fabs(z);
    double t = 1.0/(1.0+a*p);

    // 1 - 1/sqrt(2*pi) * exp( -(pow(z,2)/2) ) * ( b1*t + b2*pow(t,2.0) 
    //                                             + b3*pow(t,3.0) + b4*pow(t,4.0) 
    //                                             + b5*pow(t,5.0) )

    double b = c2*exp((-z)*(z/2.0));
    double n = ((((b5*t+b4)*t+b3)*t+b2)*t+b1)*t;

    n = 1.0 - b*n;

    if (z < 0.0) 
    {
        n = 1.0 - n;
    }

    return n;
}

// pulled from: https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/#Computer_implementations
// error is 1.15e-9
// scipy uses a different approximation from W.J. Cody published in AMS
// "Rational Chebyshev approximations for the error function" by W. J. Cody
// https://www.ams.org/journals/mcom/1969-23-107/S0025-5718-1969-0247736-4/S0025-5718-1969-0247736-4.pdf
// maximal relative error is 6e-19 to 3e-20
double inversenormalcdf(double p)
{
    double  A1 = -3.969683028665376e+01;
    double  A2 =  2.209460984245205e+02;
    double  A3 = -2.759285104469687e+02;
    double  A4 =  1.383577518672690e+02;
    double  A5 = -3.066479806614716e+01;
    double  A6 =  2.506628277459239e+00;

    double  B1 = -5.447609879822406e+01;
    double  B2 =  1.615858368580409e+02;
    double  B3 = -1.556989798598866e+02;
    double  B4 =  6.680131188771972e+01;
    double  B5 = -1.328068155288572e+01;

    double  C1 = -7.784894002430293e-03;
    double  C2 = -3.223964580411365e-01;
    double  C3 = -2.400758277161838e+00;
    double  C4 = -2.549732539343734e+00;
    double  C5 =  4.374664141464968e+00;
    double  C6 =  2.938163982698783e+00;

    double  D1  = 7.784695709041462e-03;
    double  D2 =  3.224671290700398e-01;
    double  D3 =  2.445134137142996e+00;
    double  D4 =  3.754408661907416e+00;

    double P_LOW = 0.02425;
    double P_HIGH = 0.97575;

    double x{};
    double q{};
    double r{};
    double u{};
    double e{};

    if (0 < p && p < P_LOW)
    {
        q = sqrt(-2*log(p));
        x = (((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1);
    }
  
    else if (P_LOW <= p && p <= P_HIGH)
    {
        q = p - 0.5;
        r = q*q;
        x = (((((A1*r+A2)*r+A3)*r+A4)*r+A5)*r+A6)*q /(((((B1*r+B2)*r+B3)*r+B4)*r+B5)*r+1);
    }

    else if (P_HIGH < p && p < 1) 
    {
        q = sqrt(-2*log(1-p));
        x = -(((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1);
    }

    // The relative error of the approximation has absolute value less than 
    // 1.15e−9.  One iteration of Halley’s rational method (third order) gives
    // full machine precision.
    // restricted to the range if 0 to 1 because the above correctly estimates
    // the values at 0 and 1 while the below "correction" fails.
    if (0 < p && p < 1) 
    {
        e = 0.5 * erfc(-x/sqrt(2)) - p;
        u = e * sqrt(2*M_PI) * exp(x*x/2);
        x = x - u/(1 + x*u/2);
    }
  
    return x;
}

// finds K-stage triangular design for given design parameters. The resulting 
// design is put in the vector `parameters'
void findtriangulardesign(
        double delta0,
        double delta1,
        double sigma,
        double K,
        double requiredalpha,
        double requiredbeta,
        vector<double>& parameters)
{
    double delta = delta1 - delta0;
    double i{};
    double information{};

    double psi = (2*inversenormalcdf(1-requiredalpha))/(inversenormalcdf(1-requiredalpha)+inversenormalcdf(1-requiredbeta));
    delta = psi * delta;

    double Imax=pow(sqrt(((4*pow(0.583,2))/K)+8*log((1.0/(2*requiredalpha))))-2*0.583/sqrt(K),2)/(pow(delta,2));

    int numberindividualsperstage=static_cast<int>(ceil(Imax*2*sigma*sigma/K));
    vector<double> cumulativesamplesize;
    double c,d;
    cumulativesamplesize.push_back(numberindividualsperstage);
    
    for(size_t i = 1; i < static_cast<size_t>(K); i++)
    {
        cumulativesamplesize.push_back(numberindividualsperstage+cumulativesamplesize.at(i-1));
    }

    parameters.clear();
    parameters.push_back(numberindividualsperstage);
    for(i=0;i<K;i++)
    {
        c=-(2.0/delta)*log(1.0/(2*requiredalpha))+0.583*sqrt(Imax/K)+(3*delta/4)*((i+1)/K)*Imax;
        d=(2.0/delta)*log(1.0/(2*requiredalpha))-0.583*sqrt(Imax/K)+(delta/4)*((i+1)/K)*Imax;
        information=Imax*((i+1)/K);
        parameters.push_back(c/sqrt(information));
        parameters.push_back(d/sqrt(information));
    }
}

//onestagesamplesize finds the sample size required for a one-stage trial with 
// given design parameters
double onestagesamplesize(
        double difference,
        double sigma,
        double typeIerror,
        double typeIIerror,
        double R)
{
    // ratio of smaller group to larger group
    double r = (1+R)/R;

    // z statistic for alpha
    double z_alpha = inversenormalcdf(1-typeIerror);

    // z statistic for power
    double z_power = inversenormalcdf(1-typeIIerror);
    
    return r * ((pow(sigma, 2) * pow(z_alpha + z_power, 2))/(difference*difference));
}

// calculates the information given number of individuals, delta, and sigma:
double information(double numberindividuals, double sigma)
{
  return numberindividuals/(2*sigma*sigma);
}

double expectedsamplesize(
        vector<double> phi,
        vector<double> psi,
        vector<double> parameters)
{
  size_t i;
  double expectedsamplesize=0;

  for(i=0;i<phi.size();i++)
    {
      expectedsamplesize+=(static_cast<double>(i)+1)*parameters.at(0)*(phi.at(i)+psi.at(i));
    }

  return expectedsamplesize;
}



void converthtopsi(
        vector<vector<double> >& h,
        vector<vector<double> >& z,
        vector<double>& psi,
        vector<double>& parameters,
        double delta0,
        double newdelta,
        double sigma)
{
    size_t i{};
    size_t j{};

    psi.clear();

    psi.push_back(normalcdf(parameters.at(1)-newdelta*sqrt(parameters.at(0))/sqrt(2*sigma*sigma)));

    for (i = 1; i <= z.size(); i++)
    {
        psi.push_back(0);
    
        for (j=0;j<z.at(i-1).size();j++)
        {
            psi.at(i)+=(exp((newdelta-delta0)*z.at(i-1).at(j)*sqrt(information(static_cast<double>(i)*parameters.at(0),sigma))-(pow(newdelta,2)-pow(delta0,2))*information(static_cast<double>(i)*parameters.at(0),sigma)/2)*(h.at(i-1).at(j)*(1-normalcdf((z.at(i-1).at(j)*sqrt(information(static_cast<double>(i)*parameters.at(0),sigma))-parameters.at((i+1)*2-1)*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))+newdelta*(information((static_cast<double>(i)+1)*parameters.at(0),sigma)-information(static_cast<double>(i)*parameters.at(0),sigma)))/(sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma)-information(static_cast<double>(i)*parameters.at(0),sigma)))))));
        }
    
        fflush(0);
    }

}




void converthtophi(vector<vector<double> >& h,vector<vector<double> >& z,vector<double>& phi,vector<double>& parameters,double delta0,double newdelta,double sigma)
{
  size_t i,j;

  phi.clear();

  phi.push_back(1-normalcdf(parameters.at(2)-newdelta*sqrt(parameters.at(0))/sqrt(2*sigma*sigma)));

  for(i=1;i<=z.size();i++)
    {
      phi.push_back(0);
    
      for(j=0;j<z.at(i-1).size();j++)
  {
    
      phi.at(i)+=(exp((newdelta-delta0)*z.at(i-1).at(j)*sqrt(information(static_cast<double>(i)*parameters.at(0),sigma))-(pow(newdelta,2)-pow(delta0,2))*information(static_cast<double>(i)*parameters.at(0),sigma)/2)*(h.at(i-1).at(j)*(normalcdf((z.at(i-1).at(j)*sqrt(information(static_cast<double>(i)*parameters.at(0),sigma))-parameters.at((i+1)*2)*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))+newdelta*(information((static_cast<double>(i)+1)*parameters.at(0),sigma)-information(static_cast<double>(i)*parameters.at(0),sigma)))/(sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma)-information(static_cast<double>(i)*parameters.at(0),sigma)))))));

    
    
  }
    
      fflush(0);
    }

}

//finds the delta which gives highest expected sample size for a given design

void finddeltaminimax_seq(vector<vector<double> >& h,vector<vector<double> >& z,[[maybe_unused]]vector<double>& phi,vector<double>& parameters,double delta0,double newdelta,double sigma,double *deltaminimax,double *maxen)
{

  

  double lowerdelta=delta0,middelta=newdelta,upperdelta=delta0+2*(newdelta-delta0),tempdelta,loweren,miden,upperen,tempen;
  vector<double> lowerphi,lowerpsi,midphi,midpsi,upperphi,upperpsi,tempphi,temppsi;

converthtophi(h,z,lowerphi,parameters,delta0,lowerdelta,sigma);
converthtophi(h,z,midphi,parameters,delta0,middelta,sigma);
converthtophi(h,z,upperphi,parameters,delta0,upperdelta,sigma);
converthtopsi(h,z,lowerpsi,parameters,delta0,lowerdelta,sigma);
converthtopsi(h,z,midpsi,parameters,delta0,middelta,sigma);
converthtopsi(h,z,upperpsi,parameters,delta0,upperdelta,sigma);
loweren=expectedsamplesize(lowerphi,lowerpsi,parameters);
miden=expectedsamplesize(midphi,midpsi,parameters); 
upperen=expectedsamplesize(upperphi,upperpsi,parameters);

//check whether miden is higher than both loweren and upperen

if(miden<loweren || miden<upperen)
  {
    do
      {
  lowerdelta-=(newdelta-delta0);
  upperdelta+=(newdelta-delta0);
  converthtophi(h,z,lowerphi,parameters,delta0,lowerdelta,sigma);
  converthtophi(h,z,upperphi,parameters,delta0,upperdelta,sigma);
  converthtopsi(h,z,lowerpsi,parameters,delta0,lowerdelta,sigma);
  converthtopsi(h,z,upperpsi,parameters,delta0,upperdelta,sigma);
  loweren=expectedsamplesize(lowerphi,lowerpsi,parameters);
  upperen=expectedsamplesize(upperphi,upperpsi,parameters);
  
      }
    while(miden<loweren || miden<upperen);
  }


//find delta which gives maximum expected sample size:

do
  {
  
    if((upperdelta-middelta)>(middelta-lowerdelta))
      {
  tempdelta=(upperdelta+middelta)/2;
  converthtophi(h,z,tempphi,parameters,delta0,tempdelta,sigma);
  converthtopsi(h,z,temppsi,parameters,delta0,tempdelta,sigma);
  tempen=expectedsamplesize(tempphi,temppsi,parameters);
  if(tempen<miden)
    {
      upperen=tempen;
      upperdelta=tempdelta;
      upperphi=tempphi;
      upperpsi=temppsi;
    }
  else
    {
      loweren=miden;
      lowerdelta=middelta;
      lowerphi=midphi;
      lowerpsi=midpsi;
      miden=tempen;
      middelta=tempdelta;
      midphi=tempphi;
      midpsi=temppsi;
    }
      }

    else
      {

  tempdelta=(lowerdelta+middelta)/2;
  converthtophi(h,z,tempphi,parameters,delta0,tempdelta,sigma);
  converthtopsi(h,z,temppsi,parameters,delta0,tempdelta,sigma);
  tempen=expectedsamplesize(tempphi,temppsi,parameters);

  if(tempen<miden)
    {
      loweren=tempen;
      lowerdelta=tempdelta;
      lowerphi=tempphi;
      lowerpsi=temppsi;
    }
  else
    {
      upperen=miden;
      upperdelta=middelta;
      upperphi=midphi;
      upperpsi=midpsi;
      miden=tempen;
      middelta=tempdelta;
      midphi=tempphi;
      midpsi=temppsi;
    }

      }

  }
while((middelta-lowerdelta)>1e-3 || (upperdelta-middelta)>1e-3);


*deltaminimax=middelta;
*maxen=miden;

}



//trialproperties_seq uses the method given in Section 19.2 of Jennison and Turnbull (2000) to find the probability of stopping at each stage in a sequential trial using Z-tests

void trialproperties_seq(vector<double>& parameters,double delta0,double delta1,double sigma,double *typeIerror,double *power,double *expectedsamplesize_null,double *expectedsamplesize_crd,double *worstcasedelta,double *expectedsamplesize_dm,int checkdm)
{
  //Function will find typeIerror and power for trial parameters. If checkdm==1, the worst-case scenario delta will be found together with its expected sample size. Else, both will be returned as 0

  *worstcasedelta=0;
  *expectedsamplesize_dm=0;

  size_t i,j;

  //get grid of points to use
  vector<vector<double> > x;

  vector<double> tempvector;

  for(i=0;i<(parameters.size()-1)/2-1;i++)
    {
      
      
      tempvector.clear();
      tempvector.push_back(parameters.at(i*2+1));
      for(j=1;j<=15;j++)
  {
    if(delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))-(3+4*log(16.0/static_cast<double>(j)))<parameters.at(i*2+2) && delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))-(3.0+4*log(16.0/static_cast<double>(j)))>parameters.at(i*2+1))
      {
        tempvector.push_back(delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))-(3.0+4*log(16.0/static_cast<double>(j))));
      }
    
  }
    
      for(j=16;j<=5*16;j++)
  {
    
    if(delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))-(3.0-3*(static_cast<double>(j)-16.0)/(2*16.0))<parameters.at(i*2+2) && delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))-(3.0-3*(static_cast<double>(j)-16.0)/(2*16.0))>parameters.at(i*2+1))
      {
        tempvector.push_back(delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))-(3.0-3*(static_cast<double>(j)-16.0)/(2*16.0)));
        
      }
    
  }
    
      for(j=5*16+1;j<=6*16-1;j++)
  {
    
    if(delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))+(3.0+4*log(16.0/(6*16-static_cast<double>(j))))<parameters.at(i*2+2) && delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))+(3.0+4*log(16.0/(6*16-static_cast<double>(j))))>parameters.at(i*2+1))
      {
        
        tempvector.push_back(delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))+(3.0+4*log(16.0/(6*16-static_cast<double>(j)))));
      }
    
  }

    tempvector.push_back(parameters.at(i*2+2));
    

          x.push_back(tempvector);
    

  }

  //get z's - odd numbered points are the x's, even numbered points are their midpoints 

  vector<vector<double> > z;

  for(i=0;i<x.size();i++)
    {
      tempvector.clear();
      for(j=0;j<x.at(i).size()-1;j++)
  {
    tempvector.push_back(x.at(i).at(j));
    tempvector.push_back((x.at(i).at(j)+x.at(i).at(j+1))/2);
  }
      tempvector.push_back(x.at(i).at(x.at(i).size()-1));
      z.push_back(tempvector);

    }

  //z's define the weights used in the integration:

  vector<vector<double> > weights;

  for(i=0;i<z.size();i++)
    {
      tempvector.clear();
      tempvector.push_back((z.at(i).at(2)-z.at(i).at(0))/6);
      for(j=2;j<=z.at(i).size()-1;j++)
  {
    if(j%2==0)
      {
        tempvector.push_back(4.0*(z.at(i).at(j)-z.at(i).at(j-2))/6);
      }
    else if(j%2==1)
      {
tempvector.push_back((z.at(i).at(j+1)-z.at(i).at(j-3))/6);
      }
  }
      tempvector.push_back((z.at(i).at(z.at(i).size()-1)-z.at(i).at(z.at(i).size()-3))/6);
      weights.push_back(tempvector);
      
    
    
    }

  
  //h is a matrix which has values of h at each point

  vector<vector<double> > h;
  size_t k;

  for(i=0;i<z.size();i++)
    {
      tempvector.clear();
      if(i==0)
  {
    for(j=0;j<z.at(i).size();j++)
      {
        tempvector.push_back(weights.at(i).at(j)*normalpdf(z.at(i).at(j)-delta0*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))));
      
      }
  }
      else
  {
    for(j=0;j<z.at(i).size();j++)
      {
        tempvector.push_back(0);
        for(k=0;k<z.at(i-1).size();k++)
    {
      tempvector.at(j)+=h.at(i-1).at(k)*weights.at(i).at(j)*(sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))/sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma)-information(static_cast<double>(i)*parameters.at(0),sigma)))*normalpdf((z.at(i).at(j)*sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma))-z.at(i-1).at(k)*sqrt(information(static_cast<double>(i)*parameters.at(0),sigma))-delta0*(information((static_cast<double>(i)+1)*parameters.at(0),sigma)-information(static_cast<double>(i)*parameters.at(0),sigma)))/(sqrt(information((static_cast<double>(i)+1)*parameters.at(0),sigma)-information(static_cast<double>(i)*parameters.at(0),sigma))));
  
      
    }
      }
  }
      h.push_back(tempvector);

    }


  //phi gives the probability of stopping for efficacy at each stage
  vector<double> phi,psi;
  
  converthtophi(h,z,phi,parameters,delta0,delta0,sigma);


  //calculate typeIerror by summing up phi:

  *typeIerror=0;
  for(i=0;i<phi.size();i++)
    {
      *typeIerror+=phi.at(i);
    }
converthtopsi(h,z,psi,parameters,delta0,delta0,sigma);

*expectedsamplesize_null=expectedsamplesize(phi,psi,parameters);

  converthtophi(h,z,phi,parameters,delta0,delta1,sigma);

*power=0;
  for(i=0;i<phi.size();i++)
    {
      // cout<<phi.at(i)<<" ";
      *power+=phi.at(i);
    }
  // cout<<"\n";

converthtopsi(h,z,psi,parameters,delta0,delta1,sigma);
//  for(i=0;i<psi.size();i++)
//     {
//       cout<<psi.at(i)<<" ";
    
//     }
//  cout<<"\n";


*expectedsamplesize_crd=expectedsamplesize(phi,psi,parameters);
double deltaminimax;

if(checkdm==1)
  {

finddeltaminimax_seq(h,z,phi,parameters,delta0,delta1,sigma,&deltaminimax,expectedsamplesize_dm);
*worstcasedelta=deltaminimax;
  }



    }


double functionvalue_deltaminimax(vector<double>& candidateparameters,double delta0,double delta1,double sigma,[[maybe_unused]]double K,double requiredtypeIerror,double requiredtypeIIerror,double penaltyparameter,int numberrestarts)
{


  double typeIerror,power,expectedsamplesize_null,expectedsamplesize_crd,worstcasedelta,expectedsamplesize_dm,functionvalue=0;
  
trialproperties_seq(candidateparameters,delta0,delta1,sigma,&typeIerror,&power,&expectedsamplesize_null,&expectedsamplesize_crd,&worstcasedelta,&expectedsamplesize_dm,1);

if(typeIerror>requiredtypeIerror)
{
    functionvalue+=(penaltyparameter+(typeIerror-requiredtypeIerror)/requiredtypeIerror)*penaltyparameter;
    
}

if((1-power)>requiredtypeIIerror)
{
functionvalue+=(penaltyparameter)+(((1-power)-requiredtypeIIerror)/requiredtypeIIerror)*penaltyparameter;
  
}

if((typeIerror>requiredtypeIerror || (1-power)>requiredtypeIIerror) && numberrestarts>=(-1))
  {
    
    functionvalue+=penaltyparameter/10;
  }




functionvalue+=expectedsamplesize_dm;


return(functionvalue); 
}
double checkdesignconstraints(vector<double>& design)
{
  double validdesign=1;
  size_t numberstages=(design.size()-1)/2,i;
  //check that first efficacy parameter is more than futility parameter

  if(design.at(1)>=design.at(2))
    {
      validdesign=0;
    }

  for(i=1;i<numberstages-1;i++)
    {
      if(design.at(i*2+1)>=design.at(i*2+2))
  {
    validdesign=0;
  }

      if(design.at(i*2+1)<design.at(i*2-1))
  {
    validdesign=0;
  }

      if(design.at(i*2+2)>design.at(i*2))
  {
    validdesign=0;
  }
    }

  if(design.at(numberstages*2)>design.at((numberstages-1)*2))
    {
      validdesign=0;
    }
  return validdesign;
  
}

    


double generatenormalrandomvariable()
{

  double pi=3.1415926535;
  double u1, u2;
  u1=asran();
  u2=asran();
  return(sqrt(-2*log(u1))*cos(2*pi*u2));
}


void generatecandidatestate_deltaminimax(vector<double>& currentparameters,vector<double>& candidateparameters,vector<double>& lowerranges,vector<double>& upperranges,vector<double>& parametersigmas,int fixsamplesize)
{
  //for each candidate generation, pick one stage, and perturb that stage's parameters and the sample size per stage

  size_t i;
  double u, temp;
  [[maybe_unused]] double y;
  candidateparameters=currentparameters;

  
  u=asran();

size_t numberofstages=(currentparameters.size()-1)/2;

  double temp_stagetochange=floor(static_cast<double>(numberofstages)*u);
  size_t stagetochange = static_cast<size_t>(temp_stagetochange);
  
  if(stagetochange==(numberofstages-1))
    {
    
    candidateparameters=currentparameters;
  
    //perturb sample size and last stage threshold:

    if(fixsamplesize==0)
      {
    i=0;
    do
      {
        temp=generatenormalrandomvariable();
        temp=temp*parametersigmas.at(i);
        candidateparameters.at(i)=temp+currentparameters.at(i);
      }
    while(candidateparameters.at(i)>=upperranges.at(i) || candidateparameters.at(i)<=lowerranges.at(i));
      }
    i=stagetochange*2+1;
    do
      {
        temp=generatenormalrandomvariable();
        temp=temp*parametersigmas.at(i);
        candidateparameters.at(i)=temp+currentparameters.at(i);
        candidateparameters.at(i+1)=candidateparameters.at(i);
      }
    while((candidateparameters.at(i)>=upperranges.at(i) || candidateparameters.at(i)<=lowerranges.at(i)) || checkdesignconstraints(candidateparameters)==0);
  
    }
  else
    {
      do
  {    
    candidateparameters=currentparameters;
    
    //perturb sample size and last stage threshold:
    if(fixsamplesize==0)
      {
    i=0;
    do
      {
        temp=generatenormalrandomvariable();
        temp=temp*parametersigmas.at(i);
        candidateparameters.at(i)=temp+currentparameters.at(i);
      }
    while(candidateparameters.at(i)>=upperranges.at(i) || candidateparameters.at(i)<=lowerranges.at(i));
      }
    i=stagetochange*2+1;
    do
      {
        temp=generatenormalrandomvariable();
        temp=temp*parametersigmas.at(i);
        candidateparameters.at(i)=temp+currentparameters.at(i);
        
      }
    while(candidateparameters.at(i)>=upperranges.at(i) || candidateparameters.at(i)<=lowerranges.at(i));
      i=stagetochange*2+2;
    do
      {
        temp=generatenormalrandomvariable();
        temp=temp*parametersigmas.at(i);
        candidateparameters.at(i)=temp+currentparameters.at(i);
    
      }
    while(candidateparameters.at(i)>=upperranges.at(i) || candidateparameters.at(i)<=lowerranges.at(i));
      

  }
  while(checkdesignconstraints(candidateparameters)==0);
      
    }
  

}



void simulatedannealing_deltaminimax(double delta0,double delta1,double sigma,double K,double requiredtypeIerror,double requiredpower,vector<double> &initialparameters,vector<double> lowerranges,vector<double> upperranges,vector<double>& initialparameterssigma,double initialcosttemperature,double finalparametersigma,double finalcosttemperature,int numbercandidategenerationsperrestart,int minnumberrestarts,vector<double> &finalparameters,double *finalfunctionvalue,double penaltyparameter)
{


  size_t i;
  vector<double> currentparameters=initialparameters;
  double newfunctionvalue,minimumfunctionvalue=functionvalue_deltaminimax(initialparameters,delta0,delta1,sigma,K,requiredtypeIerror,(1-requiredpower),penaltyparameter,-2),x,numbersincereduction=0,currentfunctionvalue,reductioninfunctionvalue;

  
  vector<double> parametersigmas,minimumparameters=currentparameters,candidateparameters;
  parametersigmas=initialparameterssigma;
int numberrestarts=0;
double previousrestart, candidategenerations=0,costtemperature=initialcosttemperature,rhocost=pow(finalcosttemperature/initialcosttemperature,1.0/numbercandidategenerationsperrestart),rhosigma=pow(finalparametersigma/parametersigmas.at(0),1.0/numbercandidategenerationsperrestart);
[[maybe_unused]] double minimumloss;

do
  {

generatecandidatestate_deltaminimax(currentparameters,candidateparameters,lowerranges,upperranges,parametersigmas,0);


newfunctionvalue=functionvalue_deltaminimax(candidateparameters,delta0,delta1,sigma,K,requiredtypeIerror,(1-requiredpower),penaltyparameter,numberrestarts-minnumberrestarts);

for(i=0;i<parametersigmas.size();i++)
  {
    parametersigmas.at(i)*=rhosigma;
  }

x=asran();

candidategenerations++;
if(exp(-(newfunctionvalue-currentfunctionvalue)/costtemperature)>x)
  {
  
    currentfunctionvalue=newfunctionvalue;
    costtemperature*=rhocost;
    currentparameters=candidateparameters;
    if(newfunctionvalue<minimumfunctionvalue)
      {
  minimumparameters=currentparameters;
  minimumfunctionvalue=newfunctionvalue;
  numbersincereduction=0;
  
      }
    else
      {
  numbersincereduction++;
      }

  }

else
  {
    numbersincereduction++;
  }



if((int)numbersincereduction%25==0)
  {
    currentparameters=minimumparameters;
    currentfunctionvalue=minimumfunctionvalue;
  }

if(candidategenerations>=numbercandidategenerationsperrestart)
  {
    //reset temperature
    currentparameters=minimumparameters;
    currentfunctionvalue=minimumfunctionvalue;
    costtemperature=initialcosttemperature;

    rhocost=pow(finalcosttemperature/initialcosttemperature,1.0/numbercandidategenerationsperrestart);
    parametersigmas=initialparameterssigma;
  
    rhosigma=pow(finalparametersigma/parametersigmas.at(0),1.0/numbercandidategenerationsperrestart);
    candidategenerations=0;
    numberrestarts++;
    cout<<"Restart "<<numberrestarts<<", function value = "<<minimumfunctionvalue<<"\n";
    reductioninfunctionvalue=previousrestart-minimumfunctionvalue;
    previousrestart=minimumfunctionvalue;
  }


  }
while(numberrestarts<=minnumberrestarts || reductioninfunctionvalue>0.005);

minimumparameters.at(0)=floor(minimumparameters.at(0));
minimumfunctionvalue=functionvalue_deltaminimax(minimumparameters,delta0,delta1,sigma,K,requiredtypeIerror,(1-requiredpower),penaltyparameter,1);
currentfunctionvalue=minimumfunctionvalue;
currentparameters=minimumparameters;
//repeat, but fixing samplesize
candidategenerations=0;
numberrestarts-=4;


do
  {
  
generatecandidatestate_deltaminimax(currentparameters,candidateparameters,lowerranges,upperranges,parametersigmas,1);

newfunctionvalue=functionvalue_deltaminimax(candidateparameters,delta0,delta1,sigma,K,requiredtypeIerror,(1-requiredpower),penaltyparameter,1);
// cout<<minimumfunctionvalue<<" "<<newfunctionvalue<<"\n";
for(i=0;i<parametersigmas.size();i++)
  {
    parametersigmas.at(i)*=rhosigma;
  }

x=asran();

candidategenerations++;
if(exp(-(newfunctionvalue-currentfunctionvalue)/costtemperature)>x)
  {
  
    currentfunctionvalue=newfunctionvalue;
    costtemperature*=rhocost;
    currentparameters=candidateparameters;
    if(newfunctionvalue<minimumfunctionvalue)
      {
  minimumparameters=currentparameters;
  minimumfunctionvalue=newfunctionvalue;
  numbersincereduction=0;
  

      }
    else
      {
  numbersincereduction++;
      }

  }

else
  {
    numbersincereduction++;
  }



if((int)numbersincereduction%10==0)
  {
    currentparameters=minimumparameters;
    currentfunctionvalue=minimumfunctionvalue;
  }

if(candidategenerations>=numbercandidategenerationsperrestart)
  {
    //reset temperature
    currentparameters=minimumparameters;
    currentfunctionvalue=minimumfunctionvalue;
    costtemperature=initialcosttemperature;

    rhocost=pow(finalcosttemperature/initialcosttemperature,1.0/numbercandidategenerationsperrestart);
    parametersigmas=initialparameterssigma;
  
    rhosigma=pow(finalparametersigma/parametersigmas.at(0),1.0/numbercandidategenerationsperrestart);
    candidategenerations=0;
    numberrestarts++;
    cout<<"Restart "<<numberrestarts<<", function value = "<<minimumfunctionvalue<<"\n";
    reductioninfunctionvalue=previousrestart-minimumfunctionvalue;


minimumfunctionvalue=functionvalue_deltaminimax(minimumparameters,delta0,delta1,sigma,K,requiredtypeIerror,(1-requiredpower),penaltyparameter,numberrestarts-minnumberrestarts);
    previousrestart=minimumfunctionvalue;


  }


  }
while(numberrestarts<=minnumberrestarts || reductioninfunctionvalue>0);










finalparameters=minimumparameters;
*finalfunctionvalue=minimumfunctionvalue;




}








int main(int argc, char *argv[])
{


if(argc!=8)
    {
cout<<"Usage: ./finddeltaminimaxdesign <delta0> <delta1> <sigma> <typeIerror> <power> <number of stages> <outfile>\n";
      return 0;
    }

double delta0=atof(argv[1]),delta1=atof(argv[2]),initialsigma=atof(argv[3]),requiredtypeIerror=atof(argv[4]),requiredpower=atof(argv[5]);
int K=atoi(argv[6]);
string outfilename=argv[7];

//set initial seed:

time_t seed=time(0),i;
[[maybe_unused]] int j;
  for(i=0;i<seed%10000;i++)
    {
      asran();
    }
  cout<<"Seed = "<<seed<<"\n";

//standardise problem:

  double delta=(delta1-delta0)/initialsigma,sigma=1,singlestagesamplesize=onestagesamplesize(delta,sigma,requiredtypeIerror,(1-requiredpower),1),typeIerror,power,expectedsamplesize_null,expectedsamplesize_crd,worstcasedelta,expectedsamplesize_dm;
  [[maybe_unused]] double expectedloss;
  vector<double> parameters,currentparameters,candidateparameters,lowerranges,upperranges,parametersigmas,initialparametersigmas;
  
findtriangulardesign(0,delta,sigma,K,requiredtypeIerror*49/50,(1-requiredpower),parameters);

//find trial properties of triangular design

trialproperties_seq(parameters,0,delta,sigma,&typeIerror,&power,&expectedsamplesize_null,&expectedsamplesize_crd,&worstcasedelta,&expectedsamplesize_dm,1);

//set lower ranges for parameters in simulated annealing

  lowerranges.push_back(2);
  upperranges.push_back(singlestagesamplesize);
  
  initialparametersigmas.push_back(singlestagesamplesize/5);
  for(i=0;i<K;i++)
    {
      lowerranges.push_back(-4);
      upperranges.push_back(4);
      lowerranges.push_back(-4);
      upperranges.push_back(4);
      
      initialparametersigmas.push_back(3);
      initialparametersigmas.push_back(3);

    }

  vector<double> initialparameters=parameters,finalparameters;
  double finalfunctionvalue;
  
  //carries out simulated annealing to find sample size and stopping boundaries for delta minimax design. First, the process allows n to be non-integer, searching over the sample size and stopping boundaries. After, the sample size is rounded to the nearest integer, and the stopping boundaries only are searched over.

  simulatedannealing_deltaminimax(0,delta,sigma,K,requiredtypeIerror,requiredpower,initialparameters,lowerranges,upperranges,initialparametersigmas,100,0.005,0.005,10000,5,finalparameters,&finalfunctionvalue,singlestagesamplesize);





trialproperties_seq(finalparameters,0,delta,sigma,&typeIerror,&power,&expectedsamplesize_null,&expectedsamplesize_crd,&worstcasedelta,&expectedsamplesize_dm,1);

//write results to file with specified name
ofstream outfile;
outfile.open(outfilename.c_str(),ios_base::app);
outfile<<requiredtypeIerror<<" "<<requiredpower<<" "<<K<<" "<<seed<<" "<<typeIerror<<" "<<power<<" "<<expectedsamplesize_null<<" "<<expectedsamplesize_crd<<" "<<expectedsamplesize_dm<<" ";
for(i=0;i<static_cast<int>(finalparameters.size());i++)
  {
    outfile<<finalparameters.at(static_cast<size_t>(i))<<" ";
  }
outfile<<"\n";
outfile.close();

  return 0;
}
