import React from "react";
import {connect} from "react-redux";
import Snackbar from "../../common/Snackbar";

const mapStateToProps = state => ({
    deleteState: state.video.deleteVideo,
    updateState: state.video.updateVideo,
});

const MessageBars = ({updateState, deleteState}) => (
    <React.Fragment>
        <Snackbar state={updateState} message={{success: "Process video success"}}/>
        <Snackbar state={deleteState} message={{success: "Delete video success"}}/>
    </React.Fragment>
);

export default connect(mapStateToProps)(MessageBars);
