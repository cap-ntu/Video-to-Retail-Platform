import {connect} from "react-redux";
import {MODEL_getDefaultList, MODEL_getList} from "../../../redux/actions/model/get";
import UploadCard from "./UploadCard";
import {VIDEO_upload} from "../../../redux/actions/video/post";

const mapStateToProps = state => ({
    models: state.model.models.models,
    defaultModels: state.model.defaultModels.models,
});

const mapDispatchToProps = dispatch => ({
    fetchModelList: successCallback => dispatch(MODEL_getList(successCallback)),
    fetchDefaultModelList: successCallback => dispatch(MODEL_getDefaultList(successCallback)),
    postVideo: metaData => dispatch(VIDEO_upload(metaData)),
});

export default connect(mapStateToProps, mapDispatchToProps)(UploadCard);
