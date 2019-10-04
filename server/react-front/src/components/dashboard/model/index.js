import {connect} from "react-redux";
import {MODEL_getList} from "../../../redux/actions/model/get";
import ModelManagement from "./ModelManagement";

const mapStateToProps = state => ({
    models: state.model.models.models,
});

const mapDispatchToProps = dispatch => ({
    fetchModelList: () => dispatch(MODEL_getList()),
});

export default connect(mapStateToProps, mapDispatchToProps)(ModelManagement);
