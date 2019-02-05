% octave -qf run.m <libpath> <input file> <method> [params,...]

warning("off")

printf("Running %s", program_name());

arg_list = argv();
for i = 1:nargin
  printf(" %s", arg_list{i});
endfor

printf("\n");

libpath = arg_list{1}
filename = arg_list{2}
method = arg_list{3}

other_args = {arg_list{4:end}};

addpath(genpath(libpath));

X = csvread(filename);

[mappedX, mapping] = compute_mapping(X, method, 2, other_args{:});
csvwrite(strcat(filename, '.prj'), mappedX);

printf("Finished\n");
