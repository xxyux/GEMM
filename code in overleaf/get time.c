gettimeofday(&start, NULL);
while(T--) { // T = 100
    if(...) //every version
    cudaDeviceSynchronize();
}
gettimeofday(&end,NULL);
fprintf(file, "%s, %d, %f\n", version.c_str(), N, time_diff(&start, &end)/T);// record time into csvfile