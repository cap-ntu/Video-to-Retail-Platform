import {connect} from "react-redux";
import {MODEL_create} from "../../../redux/actions/model/post";
import ModelNew from "./ModelNew";

const mapStateToProps = () => ({});

const mapDispatchToProps = dispatch => ({
    handleCreate: (model, successCallback) => dispatch(MODEL_create(model, successCallback))
});

export default connect(mapStateToProps, mapDispatchToProps)(ModelNew);
