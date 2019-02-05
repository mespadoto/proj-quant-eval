addpath(genpath('/home/mespadoto/src/drtoolbox'))

[X, labels] = generate_data('helix', 2000);
%figure, scatter3(X(:,1), X(:,2), X(:,3), 5, labels); title('Original dataset'), drawnow

no_dims = round(intrinsic_dim(X, 'MLE'));

disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);

[mappedX, mapping] = compute_mapping(X, 'PCA', no_dims);
size(mappedX)
%figure, scatter(mappedX(:,1), mappedX(:,2), 5, labels); title('Result of PCA');

[mappedX, mapping] = compute_mapping(X, 'GPLVM', no_dims, 1.0);
size(mappedX)
%figure, scatter(mappedX(:,1), mappedX(:,2), 5, labels(mapping.conn_comp)); title('Result of Laplacian Eigenmaps'); drawnow

test_toolbox()