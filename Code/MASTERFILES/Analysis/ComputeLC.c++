#include<iostream>
#include<stdlib.h>
#include<cmath> 
#include<sstream>
#include<fstream>
#include<string>
#include<iomanip>
#include<numeric>
#include<functional>
using namespace std;

const double PI  = 3.141592653589793238462;

//int j=0;
int k=0;
int Kmax;
int Kstart;
const int Tmax=100000;
const int Nmax=1;
const int Nbeadsmax=1000;

int N,Nbeads;
double Lx,Ly,Lz;

double Rg[Nmax];
double Rg2[Nmax];
double Rg4[Nmax];
double Rgave;
double Rg2ave;
double Rg4ave;

//Observables
double position[Nmax][Nbeadsmax][3];

int main(int argc, char* argv[]){
srand(time(NULL));

cout << "Write argv[1]: datain; argv[2] = dataout; argv[3]=Tmax; argv[4]:Nbeads"<< endl;
cout << "Num of files to convert:";
Kmax=int(atoi(argv[3])); 
cout << Kmax << endl;
cout << "Number of Polymers:";
N=1;
cout << N <<endl;
Nbeads=int(atoi(argv[4]));
cout<<"Number of Beads in each polymer:" << Nbeads << endl;
Kstart = int(atoi(argv[5]));


// stringstream writeFileKN;
// writeFileKN <<"KN_"<<argv[2];
// ofstream writeKN(writeFileKN.str().c_str());

// stringstream writeFileXYZ;
// writeFileXYZ <<"XYZ_"<<argv[2];
// ofstream writeXYZ(writeFileXYZ.str().c_str());

// stringstream writeFileRG;
// writeFileRG <<"RG_"<<argv[2];
// ofstream writeRG(writeFileRG.str().c_str());

// stringstream writeFileLD;
// writeFileLD <<"LD_"<<argv[2];
// ofstream writeLD(writeFileLD.str().c_str());

// stringstream writeFileLDMAX;
// writeFileLDMAX <<"MAX_"<<argv[2];
// ofstream writeLDMAX(writeFileLDMAX.str().c_str());

stringstream writeFilecurv;
writeFilecurv <<"CURVATURE_"<<argv[2];
ofstream writecurv(writeFilecurv.str().c_str());

// stringstream writeFileLCMAX;
// writeFileLCMAX <<"LCMAX_"<<argv[2];
// ofstream writeLCMAX(writeFileLCMAX.str().c_str());

stringstream writeFilecurvW;
writeFilecurvW <<"WCURVATURE_"<<argv[2];
ofstream writecurvW(writeFilecurvW.str().c_str());

// stringstream writeFileLCWMAX;
// writeFileLCWMAX <<"LCWMAX_"<<argv[2];
// ofstream writeLCWMAX(writeFileLCWMAX.str().c_str());

// stringstream writeFileCOR;
// writeFileCOR <<"COR_"<<argv[2];
// ofstream writeCOR(writeFileCOR.str().c_str());

/////////////////////////////////////////////////////
//MAIN LOOP OVER TIME!!!
////////////////////////////////////////////////////
for(k=0;k<Kmax;k++){
ifstream indata;
stringstream readFile;
readFile.str("");
readFile.clear();
long long int sum = int((k+Kstart));
if(sum==0) readFile << argv[1] <<sum;
if(sum>0)readFile << argv[1] <<sum << "0000";
indata.open(readFile.str().c_str());
cout << readFile.str().c_str()<<endl;
if(!indata){cout <<"file "<< readFile.str().c_str() << " is not there"<<endl; return 0;}

long int id,type,mol;
double num1,num2,num3;
double x,y,z;
string dummy;
long long int time;
long int Ntot;
double l1,l2;

//read 10 lines
for(int i=0;i<10;i++){
if(i==1) {
indata >> time;
time = k;
cout << "time " << time <<endl;
}
if(i==3) {
indata >> Ntot;
cout << "Ntot " << Ntot<<endl;
}
if(i==5) {
indata >> l1 >> l2;
cout << "L " << l1<< " " << l2 <<endl;
Lx = l2-l1;
cout << "Lx " << Lx <<endl;
}
if(i==6) {
indata >> l1 >> l2;
cout << "L " << l1<< " " << l2 <<endl;
Ly = l2-l1;
cout << "Ly " << Ly <<endl;
}
if(i==7) {
indata >> l1 >> l2;
cout << "L " << l1<< " " << l2 <<endl;
Lz = l2-l1;
cout << "Lz " << Lz<<endl;
}

else getline(indata,dummy);
cout << dummy<<endl;
}
 
//////////////////////////
//READ FILES
////////////////////////////
for(int n=0; n<Ntot; n++){

    indata >> id>>type>>x>>y>>z>>num1>>num2>>num3;

    int Nring = floor((double)(id-1)/(1.0*Nbeads));
    
    position[Nring][(id-1)%Nbeads][0] = (x + Lx*num1);
    position[Nring][(id-1)%Nbeads][1] = (y + Ly*num2);
    position[Nring][(id-1)%Nbeads][2] = (z + Lz*num3);

}
//////////////////////////////
//Calculating Decorrelation time
//OR Compiling xyz coordinates
////////////////////////////
// int nr=0;
// //ring
// //writeXYZ << Nbeads+1<< endl;
// for(int n=0;n<Nbeads;n++)writeXYZ << position[nr][n][0] << " " << position[nr][n][1]<<" " << position[nr][n][2] <<endl;
// writeXYZ << endl;

//////////////////////////////
//WRITE KNOT file - for KYMOKNOT
////////////////////////////
// int nr=0;
// //ring
// for(int nr=0;nr<N;nr++){
// writeKN << Nbeads+1<< endl;
// for(int n=0;n<Nbeads;n++)writeKN << position[nr][n][0] << " " << position[nr][n][1]<<" " << position[nr][n][2] <<endl;
// writeKN<<position[nr][0][0]<<" "<<position[nr][0][1]<<" " <<position[nr][0][2]<<endl;
// }
// writeKN << endl;

// /////////////////////////////
// // COMPUTE GYRATION RADIUS
// ////////////////////////////
// double rg=0;
// double com[3]={0.0,0.0,0.0};

// for(int n=0;n<Nbeads;n++){
// com[0]+=position[0][n][0];
// com[1]+=position[0][n][1];
// com[2]+=position[0][n][2];
// }
// com[0]/=Nbeads;
// com[1]/=Nbeads;
// com[2]/=Nbeads;
// for(int n=0;n<Nbeads;n++){
// for(int d=0;d<3;d++)rg+=pow(position[0][n][d]-com[d],2.0);
// }
// rg/=Nbeads;


// cout << "Rg " << rg << endl; #this is rg^2

// writeRG << time <<" "<< rg << endl;

/////////////////////////////
// COMPUTE LOCAL DENSITY FUNCTION
////////////////////////////

//parameter for spherical radius - work out what value is optimal
// double threshold=3.0;

// //calculating the local density around each monomer n to the other monomers j
// int max = 0;
// int argmax = 0;
// for(int n=0;n<Nbeads;n++){
//     //counter for the number of monomers that are within the nth monomers spherical threshold
//     int ld = 0;
//     for(int j=0;j<Nbeads;j++){
//             if (j!=n){
//                 //euclidean distance
//                 double euc = sqrt(((position[0][n][0]-position[0][j][0])*(position[0][n][0]-position[0][j][0])) + ((position[0][n][1]-position[0][j][1])*(position[0][n][1]-position[0][j][1])) + ((position[0][n][2]-position[0][j][2])*(position[0][n][2]-position[0][j][2])));
//                 if (euc<=threshold) ld++;
//         }
//     }
//     if(ld>max) {
//         max = ld;
//         argmax = n;
//     }
//     writeLD << k <<" "<< n <<" "<< ld << endl;
// }
// //file with max local density per timestep
// // writeCOR << time <<" "<< argmax <<" ";
// writeLDMAX << k <<" "<< max <<" "<< argmax << endl;
// writeLD << endl;
// writeLD << endl;

/////////////////////////////
// COMPUTE LOCAL CURVATURE FUNCTION
////////////////////////////
// double maxlc = 0.0;
// int argmaxlc = 0;
for(int n=0;n<Nbeads;n++){
    //tangents of bead i to bead i-1 and bead i to bead i+1
    double tangent1[3]={(position[0][n%Nbeads][0]-position[0][(n-1)%Nbeads][0]),(position[0][n%Nbeads][1]-position[0][(n-1)%Nbeads][1]),(position[0][n%Nbeads][2]-position[0][(n-1)%Nbeads][2])};
    double tangent2[3]={(position[0][(n+1)%Nbeads][0]-position[0][n%Nbeads][0]),(position[0][(n+1)%Nbeads][1]-position[0][n%Nbeads][1]),(position[0][(n+1)%Nbeads][2]-position[0][n%Nbeads][2])};
    if(n==0){
        //account for negative modulus error
        tangent1[0]=(position[0][n%Nbeads][0]-position[0][Nbeads-1][0]);
        tangent1[1]=(position[0][n%Nbeads][1]-position[0][Nbeads-1][1]);
        tangent1[2]=(position[0][n%Nbeads][2]-position[0][Nbeads-1][2]);
    } 
    //norm of tangent vectors
    double tangent1norm = sqrt((tangent1[0]*tangent1[0])+(tangent1[1]*tangent1[1])+(tangent1[2]*tangent1[2]));
    double tangent2norm = sqrt((tangent2[0]*tangent2[0])+(tangent2[1]*tangent2[1])+(tangent2[2]*tangent2[2]));
    //dotproduct between tangent vectors
    double dotprodval = 0;
    for(int i=0; i<3; i++)dotprodval+=(tangent1[i]*tangent2[i]);
    //local curvature of bead n
    double lc = acos(dotprodval/(tangent1norm*tangent2norm));
    writecurv << k <<" "<< n <<" "<< lc << endl;
    // if(lc>maxlc) {
    //     maxlc = lc;
    //     argmaxlc = n;
    // }
}
//file with max local curvature per timestep
// writeCOR << argmaxlc << endl;
// writeLCMAX << k <<" "<< maxlc <<" "<< argmaxlc << endl;
writecurv << endl;
writecurv << endl;

/////////////////////////////////////////////////////////
// COMPUTE LOCAL CURVATURE FUNCTION WITH WINDOW AVERAGING
/////////////////////////////////////////////////////////
// double maxlcw = 0.0;
// int argmaxlcw = 0;
int window = 3; //half of window averaging region
for(int n=0;n<Nbeads;n++){
    double lcwindow = 0;
    for(int z=n-window;z<n+window;z++){
        //tangents of bead i to bead i-1 and bead i to bead i+1
        double tangent1[3]={(position[0][z%Nbeads][0]-position[0][(z-1)%Nbeads][0]),(position[0][z%Nbeads][1]-position[0][(z-1)%Nbeads][1]),(position[0][z%Nbeads][2]-position[0][(z-1)%Nbeads][2])};
        double tangent2[3]={(position[0][(z+1)%Nbeads][0]-position[0][z%Nbeads][0]),(position[0][(z+1)%Nbeads][1]-position[0][z%Nbeads][1]),(position[0][(z+1)%Nbeads][2]-position[0][z%Nbeads][2])};
        //account for negative modulus error
        if(z==0){
            tangent1[0]=(position[0][z%Nbeads][0]-position[0][Nbeads-1][0]);
            tangent1[1]=(position[0][z%Nbeads][1]-position[0][Nbeads-1][1]);
            tangent1[2]=(position[0][z%Nbeads][2]-position[0][Nbeads-1][2]);
        } 
        else if(z==-1){
            tangent1[0]=(position[0][Nbeads+z][0]-position[0][Nbeads+(z-1)][0]);
            tangent1[1]=(position[0][Nbeads+z][1]-position[0][Nbeads+(z-1)][1]);
            tangent1[2]=(position[0][Nbeads+z][2]-position[0][Nbeads+(z-1)][2]);

            tangent2[0]=(position[0][(z+1)%Nbeads][0]-position[0][Nbeads+z][0]);
            tangent2[1]=(position[0][(z+1)%Nbeads][1]-position[0][Nbeads+z][1]);
            tangent2[2]=(position[0][(z+1)%Nbeads][2]-position[0][Nbeads+z][2]);
        }
        else if(z<-1){
            tangent1[0]=(position[0][Nbeads+z][0]-position[0][Nbeads+(z-1)][0]);
            tangent1[1]=(position[0][Nbeads+z][1]-position[0][Nbeads+(z-1)][1]);
            tangent1[2]=(position[0][Nbeads+z][2]-position[0][Nbeads+(z-1)][2]);

            tangent2[0]=(position[0][Nbeads+(z+1)][0]-position[0][Nbeads+z][0]);
            tangent2[1]=(position[0][Nbeads+(z+1)][1]-position[0][Nbeads+z][1]);
            tangent2[2]=(position[0][Nbeads+(z+1)][2]-position[0][Nbeads+z][2]);
        }
        //norm of tangent vectors
        double tangent1norm = sqrt((tangent1[0]*tangent1[0])+(tangent1[1]*tangent1[1])+(tangent1[2]*tangent1[2]));
        double tangent2norm = sqrt((tangent2[0]*tangent2[0])+(tangent2[1]*tangent2[1])+(tangent2[2]*tangent2[2]));
        //dotproduct between tangent vectors
        double dotprodval = 0;
        for(int i=0; i<3; i++)dotprodval+=(tangent1[i]*tangent2[i]);
        lcwindow += acos(dotprodval/(tangent1norm*tangent2norm));
    }
    //local curvature of bead n
    double lc = lcwindow/(2*window);
    writecurvW << k <<" "<< n <<" "<< lc << endl;
    // if(lc>maxlcw){
    //     maxlcw = lc;
    //     argmaxlcw = n;
    // }
}
// file with max local curvature per timestep
// writeCOR << argmaxlc << endl;
// writeLCWMAX << k <<" "<< maxlcw <<" "<< argmaxlcw << endl;
writecurvW << endl;
writecurvW << endl;

}//closes time (loop over k)

return 0;
}
