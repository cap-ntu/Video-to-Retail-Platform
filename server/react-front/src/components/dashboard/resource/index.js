import {connect} from 'react-redux';
import {DSH_getStatistics} from "../../../redux/actions/resource";
import ResourceManagement from "./ResourceManagement";

const mapStateToProps = state => ({
    data: state.dashboard
});

const mapDispatchToProps = dispatch => ({
    fetchStatistics: () => dispatch(DSH_getStatistics())
});

export default connect(mapStateToProps, mapDispatchToProps)(ResourceManagement);