namespace py example

service Calculator {
    i32 add(1:i32 num1, 2:i32 num2),
}

struct FeaturesResult {
    1: binary features,
    2: binary boxes
}

service Features {
    FeaturesResult get_features(1:binary jpg_img),
}