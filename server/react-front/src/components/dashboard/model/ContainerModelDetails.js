import {connect} from "react-redux";
import ModelDetails from "./ModelDetails";
import {MODEL_delete, MODEL_getSingle} from "../../../redux/actions/model";

const mapStateToProps = state => ({
    model: state.model.singleModel.model,
    deleteState: state.model.deleteModel,
});

const mapDispatchToProps = dispatch => ({
    fetchModel: id => dispatch(MODEL_getSingle(id)),
    deleteModel: (id, successCallback) => dispatch(MODEL_delete(id, successCallback)),
});

export default connect(mapStateToProps, mapDispatchToProps)(ModelDetails);
