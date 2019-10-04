import {connect} from "react-redux";
import UserManagement from "./UserManagement";
import {USER_create, USER_delete, USER_getList, USER_update} from "../../../redux/actions/user";

const mapStateToProps = state => ({
    users: state.user.userList.users,
    nameList: state.user.userList.nameList,
});

const mapDispatchToProps = dispatch => ({
    fetchUserList: () => dispatch(USER_getList()),
    handleNewUser: (user, successCallback) => dispatch(USER_create(user, successCallback)),
    handleUpdateUser: (user, successCallback) => dispatch(USER_update(user, successCallback)),
    handleDeleteUser: (id, successCallback) => dispatch(USER_delete(id, successCallback)),
});

export default connect(mapStateToProps, mapDispatchToProps)(UserManagement);
