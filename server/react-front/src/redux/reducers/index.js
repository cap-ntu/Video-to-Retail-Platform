import combineReducers from 'redux/src/combineReducers';
import dashboard from "./dashboard";
import video from "./video/video";
import model from "./model/model";
import user from "./user/user";
import product from "./product/product";


export default combineReducers({dashboard, video, user, model, product});
